import xml.etree.ElementTree as ET
from urllib import parse
import requests
import traceback
import AES256Util
import datetime
import re
import json


# ============================================================
# ✅ 전역 변수 선언
# ============================================================
refreshToken = None
accessToken = None
KISTI_REFRESH_TOKEN = None
KISTI_ACCESS_TOKEN = None


# ============================================================
# ✅ Access Token 및 Refresh Token 발급
# ============================================================
def createToken(mac_address=None, client_id=None, key_value=None):
    """
    외부에서 mac_address, client_id, key_value를 전달받아 토큰 생성.
    전달하지 않으면 기존 전역 설정을 사용.
    """
    global refreshToken, accessToken, KISTI_REFRESH_TOKEN, KISTI_ACCESS_TOKEN

    # ✅ 기본값 설정 (외부 ipynb에서 넘기지 않은 경우)
    mac_address = mac_address or globals().get("MAC_address", "")
    client_id = client_id or globals().get("clientID", "")
    key_value = key_value or globals().get("key", "")

    if not (mac_address and client_id and key_value):
        raise ValueError("MAC 주소, client_id, key 중 누락된 값이 있습니다. createToken(mac, id, key) 형태로 입력하세요.")

    # 현재 시각 + MAC 주소 암호화
    time = ''.join(re.findall(r"\d", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    plain_txt = json.dumps({"datetime": time, "mac_address": mac_address}).replace(" ", "")

    encryption = AES256Util.AESTestClass(plain_txt, key_value)
    encrypted_txt = encryption.encrypt()
    target_URL = f"https://apigateway.kisti.re.kr/tokenrequest.do?client_id={client_id}&accounts={encrypted_txt}"

    try:
        response = requests.get(target_URL)
        response.raise_for_status()
        json_object = json.loads(response.text)

        # ✅ 토큰 저장
        refreshToken = json_object.get('refresh_token', None)
        accessToken = json_object.get('access_token', None)

        # ✅ 글로벌 변수에 KISTI용으로도 저장
        KISTI_REFRESH_TOKEN = refreshToken
        KISTI_ACCESS_TOKEN = accessToken

        print('✅ Refresh Token, Access Token 발행 완료')
        print('KISTI_REFRESH_TOKEN:', KISTI_REFRESH_TOKEN)
        print('KISTI_ACCESS_TOKEN:', KISTI_ACCESS_TOKEN)

        return {"refresh_token": refreshToken, "access_token": accessToken}

    except Exception:
        traceback.print_exc()
        return None


# ============================================================
# ✅ Access Token 재발급
# ============================================================
def getAccessToken(client_id=None):
    """
    이미 발급받은 refreshToken으로 Access Token을 새로 갱신.
    외부에서 client_id를 넘길 수도 있음.
    """
    global refreshToken, accessToken, KISTI_ACCESS_TOKEN

    client_id = client_id or globals().get("clientID", "")
    if not (refreshToken and client_id):
        raise ValueError("refreshToken 또는 client_id가 없습니다. 먼저 createToken()을 실행하세요.")

    target_URL = f"https://apigateway.kisti.re.kr/tokenrequest.do?refreshToken={refreshToken}&client_id={client_id}"

    try:
        response = requests.get(target_URL)
        response.raise_for_status()

        if 'errorCode' in response.text:
            print("⚠ Refresh Token 만료됨. createToken()으로 재발급 필요.")
            return None
        else:
            json_object = json.loads(response.text)
            accessToken = json_object.get('access_token', None)
            KISTI_ACCESS_TOKEN = accessToken  # ✅ 글로벌 갱신
            print('🔄 Access Token 재발행 완료')
            print('KISTI_ACCESS_TOKEN:', KISTI_ACCESS_TOKEN)
            return accessToken

    except Exception:
        traceback.print_exc()
        return None
