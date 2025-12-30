# Fuseki MCP Server

Apache Jena Fuseki용 Model Context Protocol (MCP) 서버입니다.
Claude Desktop과 통신하여 SPARQL 쿼리를 실행하고 SAP 회계 데이터를 조회할 수 있습니다.

## 설치

```bash
cd /Users/eric2020/Documents/ws_python/sap_snowflake_to_jena/mcp_server_fuseki

# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 실행

```bash
# 가상환경 활성화
source venv/bin/activate

# MCP 서버 시작
python server.py
```

## Claude Desktop 설정

Claude Desktop의 설정 파일에 다음을 추가하세요:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "fuseki": {
      "command": "/Users/eric2020/Documents/ws_python/sap_snowflake_to_jena/mcp_server_fuseki/venv/bin/python",
      "args": [
        "/Users/eric2020/Documents/ws_python/sap_snowflake_to_jena/mcp_server_fuseki/server.py"
      ]
    }
  }
}
```

## 사용 가능한 도구

### 1. sparql_query
SPARQL SELECT 쿼리 실행

```
Tool: sparql_query
Arguments: {"query": "SELECT * WHERE { ?s ?p ?o } LIMIT 10"}
```

### 2. sparql_update
SPARQL UPDATE 쿼리 실행

```
Tool: sparql_update
Arguments: {"query": "INSERT DATA { ... }"}
```

### 3. get_dataset_info
데이터셋 통계 정보 조회

```
Tool: get_dataset_info
```

### 4. count_triples
전체 트리플 개수 조회

```
Tool: count_triples
```

### 5. find_documents
SAP 회계 전표 검색

```
Tool: find_documents
Arguments: {
  "company_code": "1000",
  "fiscal_year": "2025",
  "document_type": "AB",
  "limit": 10
}
```

### 6. get_document_details
특정 전표의 상세 정보 조회

```
Tool: get_document_details
Arguments: {
  "company_code": "1000",
  "fiscal_year": "2025",
  "document_number": "0100106721"
}
```

## 리소스

- `fuseki://sap/stats`: 데이터셋 통계
- `fuseki://sap/namespaces`: 네임스페이스 목록

## 예제 쿼리

### 월별 전표 개수
```sparql
PREFIX sap: <http://fnf.co.kr/ontology/sap#>

SELECT ?period (COUNT(?doc) as ?count)
WHERE {
  ?doc a sap:AccountingDocument .
  ?doc sap:period ?period .
}
GROUP BY ?period
ORDER BY ?period
```

### 회사별 총 금액
```sparql
PREFIX sap: <http://fnf.co.kr/ontology/sap#>

SELECT ?company (SUM(?amount) as ?total)
WHERE {
  ?doc a sap:AccountingDocument .
  ?doc sap:companyCode ?company .
  ?doc sap:hasLineItem ?line .
  ?line sap:amountLocalCurrency ?amount .
}
GROUP BY ?company
```

## 설정

서버 설정은 `server.py` 상단에서 수정할 수 있습니다:

```python
FUSEKI_BASE_URL = "http://localhost:3030"
DATASET_NAME = "sap"
```

## 문제 해결

### Fuseki 서버 연결 실패
- Fuseki 서버가 실행 중인지 확인: `curl http://localhost:3030/$/ping`
- 데이터셋이 존재하는지 확인: `curl http://localhost:3030/$/datasets`

### MCP 서버 로그 확인
서버는 표준 출력으로 로그를 출력합니다. Claude Desktop 로그에서 확인할 수 있습니다.
# poc_knowledge_mcp_server
