# OG API - Detailed Documentation

## Overview

The `og_api.py` is a FastAPI application that provides a combined API endpoint (`/OG_data`) which orchestrates calls to two external APIs (V6 and Zylo) in sequence. It serves as a middleware layer that extracts data from the V6 API response and uses it to call the Zylo API.

## API Endpoint

### `/OG_data` (POST)

- **Method**: POST
- **Content-Type**: application/json
- **Port**: 8800
- **Host**: 0.0.0.0

## Input Format

### Request Structure

```json
{
  "iris_id": "string"
}
```

### Input Source

- **Origin**: HTTP POST request from client applications
- **Format**: JSON body with `iris_id` field
- **Validation**: Pydantic model ensures `iris_id` is a string type

### Example Input

```json
{
  "iris_id": "IRIS123456789"
}
```

## Processing Flow and Checkpoints

### Checkpoint 1: Request Validation

- **Location**: Line 16
- **Logic**: FastAPI validates request against `CombinedRequest` Pydantic model
- **Validation Rules**:
  - `iris_id` must be present and of type string
- **Failure Response**: HTTP 422 Unprocessable Entity

### Checkpoint 2: V6 API Call Preparation

- **Location**: Lines 20-28
- **Logic**:
  - Retrieve V6 API URL from `v6_api` environment variable
  - Retrieve authentication token from `token` environment variable
  - Construct request payload with `iris_id`
- **Environment Variables Required**:
  - `v6_api`: V6 API endpoint URL
  - `token`: Authentication token for V6 API

### Checkpoint 3: V6 API Execution

- **Location**: Lines 30-35
- **Logic**:
  - Make asynchronous POST request to V6 API
  - Validate HTTP response status
  - Parse JSON response
- **Error Handling**: HTTP status errors raise HTTPException with original error details

### Checkpoint 4: ID Extraction

- **Location**: Lines 37-43
- **Logic**:
  - Check if `jira_fields` exists in V6 response
  - Extract `customer_id` from `jira_fields.cid`
  - Extract `entitlement_id` from `jira_fields.eid`
- **Extraction Rules**:
  - If `jira_fields` is missing, both IDs remain `None`
  - If `cid` or `eid` are missing, corresponding ID remains `None`

### Checkpoint 5: Zylo API Call Preparation

- **Location**: Lines 45-60
- **Logic**:
  - Retrieve Zylo endpoint from `zylo_endpoint` environment variable
  - Retrieve Bearer token from `Bearer_Token` environment variable
  - Construct authorization header
  - Build payload with extracted IDs (only if available)
- **Environment Variables Required**:
  - `zylo_endpoint`: Zylo API endpoint URL
  - `Bearer_Token`: Bearer token for Zylo API

### Checkpoint 6: Zylo API Execution

- **Location**: Lines 62-64
- **Logic**:
  - Make asynchronous POST request to Zylo API
  - Validate HTTP response status
  - Parse JSON response
- **Payload Construction**:
  - Only includes `customerId` if `customer_id` was extracted
  - Only includes `entitlementlId` if `entitlement_id` was extracted

### Checkpoint 7: Response Composition

- **Location**: Lines 66-74
- **Logic**: Combine all data into unified response structure

## Output Format

### Success Response Structure

```json
{
  "v6_response": {
    // Complete V6 API response data
  },
  "zylo_response": {
    // Complete Zylo API response data
  },
  "extracted_ids": {
    "customer_id": "string or null",
    "entitlement_id": "string or null"
  }
}
```

### Error Response Structure

```json
{
  "detail": "Error description"
}
```

### HTTP Status Codes

- **200**: Success
- **422**: Validation error (invalid input format)
- **External API error codes**: Propagated from external APIs (400, 401, 403, 404, 500, etc.)
- **500**: Internal server error

## Environment Variables Configuration

### Required Environment Variables

```bash
v6_api=https://v6-api-endpoint.com/api
token=your_v6_authentication_token
zylo_endpoint=https://zylo-api-endpoint.com/api
Bearer_Token=your_zylo_bearer_token
```

### Configuration File

- Uses `.env` file in the same directory
- Loaded automatically via `load_dotenv()`

## Test Scenarios

### Test Scenario 1: Successful Flow

**Input**:

```json
{
  "iris_id": "IRIS123456789"
}
```

**Expected V6 Response**:

```json
{
  "jira_fields": {
    "cid": "CUST001",
    "eid": "ENT001"
  },
  "other_v6_data": "..."
}
```

**Expected Zylo Payload**:

```json
{
  "customerId": "CUST001",
  "entitlementlId": "ENT001"
}
```

**Expected Final Response**:

```json
{
  "v6_response": { "jira_fields": { "cid": "CUST001", "eid": "ENT001" } },
  "zylo_response": { "zylo_data": "..." },
  "extracted_ids": {
    "customer_id": "CUST001",
    "entitlement_id": "ENT001"
  }
}
```

### Test Scenario 2: Missing Jira Fields

**Input**:

```json
{
  "iris_id": "IRIS123456789"
}
```

**Expected V6 Response**:

```json
{
  "some_data": "no jira_fields here"
}
```

**Expected Zylo Payload**:

```json
{}
```

**Expected Final Response**:

```json
{
  "v6_response": { "some_data": "no jira_fields here" },
  "zylo_response": { "zylo_data": "..." },
  "extracted_ids": {
    "customer_id": null,
    "entitlement_id": null
  }
}
```

### Test Scenario 3: Partial ID Extraction

**Input**:

```json
{
  "iris_id": "IRIS123456789"
}
```

**Expected V6 Response**:

```json
{
  "jira_fields": {
    "cid": "CUST001"
    // eid is missing
  }
}
```

**Expected Zylo Payload**:

```json
{
  "customerId": "CUST001"
  // entitlementlId not included
}
```

**Expected Final Response**:

```json
{
  "v6_response": { "jira_fields": { "cid": "CUST001" } },
  "zylo_response": { "zylo_data": "..." },
  "extracted_ids": {
    "customer_id": "CUST001",
    "entitlement_id": null
  }
}
```

### Test Scenario 4: Input Validation Error

**Input**:

```json
{
  // missing iris_id
}
```

**Expected Response**:

```json
{
  "detail": [
    {
      "loc": ["body", "iris_id"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**HTTP Status**: 422

### Test Scenario 5: V6 API Error

**Input**:

```json
{
  "iris_id": "INVALID_IRIS_ID"
}
```

**Expected V6 Response**: HTTP 404 Not Found

**Expected Final Response**:

```json
{
  "detail": "External API error: 404 Not Found"
}
```

**HTTP Status**: 404

### Test Scenario 6: Zylo API Error

**Input**:

```json
{
  "iris_id": "IRIS123456789"
}
```

**Expected V6 Response**: Success with IDs extracted
**Expected Zylo Response**: HTTP 401 Unauthorized

**Expected Final Response**:

```json
{
  "detail": "External API error: 401 Unauthorized"
}
```

**HTTP Status**: 401

## Technical Implementation Details

### Dependencies

- **FastAPI**: Web framework
- **httpx**: Async HTTP client
- **pydantic**: Data validation
- **python-dotenv**: Environment variable management
- **uvicorn**: ASGI server

### Async Processing

- Uses `async/await` for non-blocking HTTP requests
- Single `httpx.AsyncClient` instance for both API calls
- Error handling preserves original error messages

### Security Considerations

- Authentication tokens stored in environment variables
- No sensitive data logged or exposed in error messages
- Input validation prevents malformed requests

## Deployment Instructions

### Local Development

```bash
# Install dependencies
pip install fastapi httpx python-dotenv uvicorn

# Set environment variables in .env file
echo "v6_api=https://your-v6-api.com" > .env
echo "token=your_token" >> .env
echo "zylo_endpoint=https://your-zylo-api.com" >> .env
echo "Bearer_Token=your_bearer_token" >> .env

# Run the server
python og_api_clean.py
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8800
CMD ["python", "og_api_clean.py"]
```

### Health Check

- Endpoint: `GET /health` (if implemented)
- Server runs on `0.0.0.0:8800`
- Readiness: Server is listening on port 8800

## Monitoring and Logging

### Recommended Monitoring Points

1. Request/response latency for both external APIs
2. Error rates for V6 and Zylo API calls
3. ID extraction success/failure rates
4. Overall endpoint response times

### Logging Recommendations

- Log incoming requests (without sensitive data)
- Log API call timestamps and durations
- Log extraction results (success/failure)
- Log errors with appropriate severity levels
