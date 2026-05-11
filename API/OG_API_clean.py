from fastapi import FastAPI, HTTPException
import httpx
import json
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

class CombinedRequest(BaseModel):
    iris_id: str

@app.post("/OG_data")
async def combined_api(request: CombinedRequest):
    """
    Combined endpoint that calls V6 API first, then Zylo API with extracted IDs
    """
    # Step 1: Call V6 API
    v6_url = os.getenv('v6_api')
    v6_headers = {
        'token': os.getenv('token'),
        'Content-Type': 'application/json'
    }
    v6_data = {
        "iris_id": request.iris_id
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # Call V6 API
            v6_response = await client.post(v6_url, headers=v6_headers, json=v6_data)
            v6_response.raise_for_status()
            v6_data = v6_response.json()
            
            # Extract customer_id and entitlement_id from jira_fields
            customer_id = None
            entitlement_id = None
            if 'jira_fields' in v6_data:
                jira_fields = v6_data['jira_fields']
                customer_id = jira_fields.get('cid')
                entitlement_id = jira_fields.get('eid')
            
            # Step 2: Call Zylo API with extracted IDs
            zylo_endpoint = os.getenv("zylo_endpoint")
            bearer_token = os.getenv("Bearer_Token")
            
            zylo_headers = {
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json"
            }
            
            zylo_data = {}
            
            # Add customer_id and entitlement_id to payload if available
            if customer_id:
                zylo_data["customerId"] = customer_id
            if entitlement_id:
                zylo_data["entitlementlId"] = entitlement_id
            
            zylo_response = await client.post(zylo_endpoint, headers=zylo_headers, json=zylo_data)
            zylo_response.raise_for_status()
            zylo_data = zylo_response.json()
            
            # Return combined response
            return {
                "v6_response": v6_data,
                "zylo_response": zylo_data,
                "extracted_ids": {
                    "customer_id": customer_id,
                    "entitlement_id": entitlement_id
                }
            }
            
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"External API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)
