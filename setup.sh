#!/bin/bash

# Fuseki MCP Server 설정 스크립트

echo "========================================="
echo "Fuseki MCP Server 설정"
echo "========================================="
echo ""

# 현재 디렉토리로 이동
cd "$(dirname "$0")"

# 가상환경 생성
echo "1. 가상환경 생성 중..."
python3 -m venv venv

# 가상환경 활성화
echo "2. 가상환경 활성화 중..."
source venv/bin/activate

# 의존성 설치
echo "3. 의존성 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "========================================="
echo "설치 완료!"
echo "========================================="
echo ""
echo "다음 단계:"
echo ""
echo "1. Claude Desktop 설정 파일 편집:"
echo "   ~/Library/Application Support/Claude/claude_desktop_config.json"
echo ""
echo "2. 다음 내용 추가:"
echo ""
cat << 'EOF'
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
EOF
echo ""
echo "3. Claude Desktop 재시작"
echo ""
echo "4. MCP 서버 테스트:"
echo "   source venv/bin/activate"
echo "   python server.py"
echo ""
