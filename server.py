#!/usr/bin/env python3
"""
Fuseki MCP Server
Apache Jena Fuseki용 Model Context Protocol 서버
Claude Desktop과 통신하여 SPARQL 쿼리를 실행합니다.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any, Sequence
from urllib.parse import urljoin

import httpx
from cachetools import TTLCache
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fuseki-mcp-server")

# Fuseki 서버 설정 (환경변수로 오버라이드 가능)
FUSEKI_BASE_URL = os.environ.get("FUSEKI_BASE_URL", "http://localhost:3030")
DEFAULT_DATASETS = ["fnco-sap-fi", "fnf-product", "fnf-hr"]
DATASETS = os.environ.get("FUSEKI_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
DEFAULT_DATASET = os.environ.get("FUSEKI_DEFAULT_DATASET", DATASETS[0])

# 캐시 설정
CACHE_TTL_NORMAL = 300  # 일반 쿼리: 5분
CACHE_TTL_AGGREGATE = 1800  # 집계 쿼리: 30분
CACHE_MAX_SIZE = 1000  # 최대 캐시 항목 수

# 캐시 인스턴스 생성
query_cache: TTLCache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_NORMAL)
aggregate_cache: TTLCache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL_AGGREGATE)

# 집계 쿼리 패턴 (대소문자 무시)
AGGREGATE_PATTERN = re.compile(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP\s+BY)\b', re.IGNORECASE)

def get_query_endpoint(dataset: str) -> str:
    return f"{FUSEKI_BASE_URL}/{dataset}/query"

def is_aggregate_query(query: str) -> bool:
    """집계 쿼리 여부 판별 (COUNT, SUM, AVG, MIN, MAX, GROUP BY 포함 시)"""
    return bool(AGGREGATE_PATTERN.search(query))

def get_cache_key(query: str, dataset: str) -> str:
    """쿼리와 데이터셋을 기반으로 캐시 키 생성"""
    normalized_query = ' '.join(query.split())  # 공백 정규화
    key_string = f"{dataset}:{normalized_query}"
    return hashlib.sha256(key_string.encode()).hexdigest()

# MCP 서버 생성
app = Server("fuseki-mcp-server")


@app.list_resources()
async def list_resources() -> list[Resource]:
    """사용 가능한 리소스 목록 반환"""
    resources = []
    for dataset in DATASETS:
        resources.extend([
            Resource(
                uri=f"fuseki://{dataset}/stats",
                name=f"{dataset} Statistics",
                mimeType="application/json",
                description=f"Statistics for {dataset} dataset"
            ),
            Resource(
                uri=f"fuseki://{dataset}/namespaces",
                name=f"{dataset} Namespaces",
                mimeType="application/json",
                description=f"List of namespaces in {dataset} dataset"
            )
        ])
    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    """리소스 읽기"""

    # URI에서 데이터셋 이름 추출
    for dataset in DATASETS:
        if uri == f"fuseki://{dataset}/stats":
            # 통계 쿼리
            query = """
            SELECT
                (COUNT(*) as ?totalTriples)
                (COUNT(DISTINCT ?s) as ?subjects)
                (COUNT(DISTINCT ?p) as ?predicates)
                (COUNT(DISTINCT ?o) as ?objects)
            WHERE {
                ?s ?p ?o
            }
            """
            result = await execute_sparql_query(query, dataset)
            return json.dumps(result, indent=2)

        elif uri == f"fuseki://{dataset}/namespaces":
            # 네임스페이스 쿼리
            query = """
            SELECT DISTINCT ?namespace
            WHERE {
                ?s ?p ?o .
                BIND(REPLACE(STR(?s), "(.*[/#])[^/#]*$", "$1") AS ?namespace)
            }
            LIMIT 100
            """
            result = await execute_sparql_query(query, dataset)
            return json.dumps(result, indent=2)

    return json.dumps({"error": f"Unknown resource: {uri}"})


@app.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구 목록 반환"""
    dataset_enum = DATASETS
    return [
        Tool(
            name="list_datasets",
            description="List all available Fuseki datasets",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="sparql_query",
            description="Execute a SPARQL SELECT query on the Fuseki dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SPARQL SELECT query to execute"
                    },
                    "dataset": {
                        "type": "string",
                        "enum": dataset_enum,
                        "description": f"Dataset to query (default: {DEFAULT_DATASET})"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_dataset_info",
            description="Get information about a Fuseki dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "enum": dataset_enum,
                        "description": f"Dataset to get info (default: {DEFAULT_DATASET})"
                    }
                }
            }
        ),
        Tool(
            name="count_triples",
            description="Count total number of triples in a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "enum": dataset_enum,
                        "description": f"Dataset to count (default: {DEFAULT_DATASET})"
                    }
                }
            }
        ),
        Tool(
            name="find_documents",
            description="Find SAP accounting documents by criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "company_code": {
                        "type": "string",
                        "description": "Company code (optional)"
                    },
                    "fiscal_year": {
                        "type": "string",
                        "description": "Fiscal year (optional)"
                    },
                    "document_type": {
                        "type": "string",
                        "description": "Document type (optional)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)"
                    }
                }
            }
        ),
        Tool(
            name="get_document_details",
            description="Get detailed information about a specific accounting document",
            inputSchema={
                "type": "object",
                "properties": {
                    "company_code": {
                        "type": "string",
                        "description": "Company code"
                    },
                    "fiscal_year": {
                        "type": "string",
                        "description": "Fiscal year"
                    },
                    "document_number": {
                        "type": "string",
                        "description": "Document number"
                    }
                },
                "required": ["company_code", "fiscal_year", "document_number"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """도구 실행"""

    try:
        if name == "list_datasets":
            return [TextContent(
                type="text",
                text=json.dumps({
                    "datasets": DATASETS,
                    "default": DEFAULT_DATASET
                }, indent=2, ensure_ascii=False)
            )]

        elif name == "sparql_query":
            query = arguments.get("query")
            dataset = arguments.get("dataset", DEFAULT_DATASET)
            result = await execute_sparql_query(query, dataset)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]

        elif name == "get_dataset_info":
            dataset = arguments.get("dataset", DEFAULT_DATASET)
            info = await get_dataset_info(dataset)
            return [TextContent(
                type="text",
                text=json.dumps(info, indent=2, ensure_ascii=False)
            )]

        elif name == "count_triples":
            dataset = arguments.get("dataset", DEFAULT_DATASET)
            count = await count_triples(dataset)
            return [TextContent(
                type="text",
                text=f"Total triples in {dataset}: {count:,}"
            )]

        elif name == "find_documents":
            results = await find_documents(
                company_code=arguments.get("company_code"),
                fiscal_year=arguments.get("fiscal_year"),
                document_type=arguments.get("document_type"),
                limit=arguments.get("limit", 10)
            )
            return [TextContent(
                type="text",
                text=json.dumps(results, indent=2, ensure_ascii=False)
            )]

        elif name == "get_document_details":
            details = await get_document_details(
                company_code=arguments["company_code"],
                fiscal_year=arguments["fiscal_year"],
                document_number=arguments["document_number"]
            )
            return [TextContent(
                type="text",
                text=json.dumps(details, indent=2, ensure_ascii=False)
            )]

        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


# Helper Functions

async def execute_sparql_query(query: str, dataset: str = DEFAULT_DATASET) -> dict:
    """SPARQL SELECT 쿼리 실행 (캐시 적용)"""
    cache_key = get_cache_key(query, dataset)
    is_aggregate = is_aggregate_query(query)

    # 적절한 캐시 선택
    cache = aggregate_cache if is_aggregate else query_cache
    cache_type = "aggregate" if is_aggregate else "normal"

    # 캐시 히트 확인
    if cache_key in cache:
        logger.info(f"Cache HIT ({cache_type}): {cache_key[:16]}... [dataset={dataset}]")
        return cache[cache_key]

    logger.info(f"Cache MISS ({cache_type}): {cache_key[:16]}... [dataset={dataset}]")

    # 캐시 미스: 실제 쿼리 실행
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            get_query_endpoint(dataset),
            headers={"Content-Type": "application/sparql-query"},
            content=query
        )
        response.raise_for_status()
        result = response.json()

    # 결과 캐시에 저장
    cache[cache_key] = result
    logger.info(f"Cached ({cache_type}, TTL={CACHE_TTL_AGGREGATE if is_aggregate else CACHE_TTL_NORMAL}s): {cache_key[:16]}...")

    return result


async def get_dataset_info(dataset: str = DEFAULT_DATASET) -> dict:
    """데이터셋 정보 조회"""
    query = """
    PREFIX sap: <http://fnf.co.kr/ontology/sap#>

    SELECT
        (COUNT(DISTINCT ?doc) as ?documentCount)
        (COUNT(DISTINCT ?line) as ?lineItemCount)
        (COUNT(*) as ?totalTriples)
    WHERE {
        {
            ?doc a sap:AccountingDocument
        } UNION {
            ?line a sap:LineItem
        } UNION {
            ?s ?p ?o
        }
    }
    """
    result = await execute_sparql_query(query, dataset)

    if result.get("results", {}).get("bindings"):
        binding = result["results"]["bindings"][0]
        return {
            "dataset": dataset,
            "documents": int(binding.get("documentCount", {}).get("value", 0)),
            "lineItems": int(binding.get("lineItemCount", {}).get("value", 0)),
            "totalTriples": int(binding.get("totalTriples", {}).get("value", 0))
        }
    return {"dataset": dataset}


async def count_triples(dataset: str = DEFAULT_DATASET) -> int:
    """전체 트리플 개수 조회"""
    query = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }"
    result = await execute_sparql_query(query, dataset)

    if result.get("results", {}).get("bindings"):
        return int(result["results"]["bindings"][0]["count"]["value"])
    return 0


async def find_documents(company_code: str = None, fiscal_year: str = None,
                        document_type: str = None, limit: int = 10) -> dict:
    """회계 전표 검색"""

    filters = []
    if company_code:
        filters.append(f'?doc sap:companyCode "{company_code}"')
    if fiscal_year:
        filters.append(f'?doc sap:fiscalYear "{fiscal_year}"')
    if document_type:
        filters.append(f'?doc sap:documentType "{document_type}"')

    filter_clause = " .\n        ".join(filters) if filters else ""

    query = f"""
    PREFIX sap: <http://fnf.co.kr/ontology/sap#>

    SELECT ?doc ?companyCode ?fiscalYear ?docNumber ?docType ?postingDate
    WHERE {{
        ?doc a sap:AccountingDocument .
        ?doc sap:companyCode ?companyCode .
        ?doc sap:fiscalYear ?fiscalYear .
        ?doc sap:documentNumber ?docNumber .
        ?doc sap:documentType ?docType .
        ?doc sap:postingDate ?postingDate .
        {filter_clause}
    }}
    ORDER BY DESC(?postingDate)
    LIMIT {limit}
    """

    return await execute_sparql_query(query)


async def get_document_details(company_code: str, fiscal_year: str,
                               document_number: str) -> dict:
    """전표 상세 정보 조회"""

    query = f"""
    PREFIX sap: <http://fnf.co.kr/ontology/sap#>
    PREFIX inst: <http://fnf.co.kr/data/sap/>

    SELECT ?property ?value
    WHERE {{
        inst:document/{company_code}/{fiscal_year}/{document_number} ?property ?value .
    }}
    """

    result = await execute_sparql_query(query)

    # 라인 아이템도 조회
    lines_query = f"""
    PREFIX sap: <http://fnf.co.kr/ontology/sap#>
    PREFIX inst: <http://fnf.co.kr/data/sap/>

    SELECT ?lineItem ?lineNumber ?account ?amount ?debitCredit
    WHERE {{
        inst:document/{company_code}/{fiscal_year}/{document_number} sap:hasLineItem ?lineItem .
        ?lineItem sap:lineItemNumber ?lineNumber .
        OPTIONAL {{ ?lineItem sap:glAccount ?account }}
        OPTIONAL {{ ?lineItem sap:amountLocalCurrency ?amount }}
        OPTIONAL {{ ?lineItem sap:debitCreditIndicator ?debitCredit }}
    }}
    ORDER BY ?lineNumber
    """

    lines_result = await execute_sparql_query(lines_query)

    return {
        "document": result,
        "lineItems": lines_result
    }


async def main():
    """MCP 서버 시작"""
    logger.info("Starting Fuseki MCP Server...")
    logger.info(f"Fuseki endpoint: {FUSEKI_BASE_URL}")
    logger.info(f"Available datasets: {', '.join(DATASETS)}")
    logger.info(f"Default dataset: {DEFAULT_DATASET}")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
