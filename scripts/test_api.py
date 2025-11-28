import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load .env from current directory
load_dotenv('.env')

print('='*70)
print('DETAILED API KEY TESTING')
print('='*70)

# 1. OpenRouter API
print('\n1. OPENROUTER API TEST')
print('-'*70)
api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
print(f'API Key found: {"Yes" if api_key else "No"}')
if api_key:
    print(f'API Key (first 15 chars): {api_key[:15]}...')
    try:
        client = OpenAI(
            base_url='https://openrouter.ai/api/v1',
            api_key=api_key
        )
        response = client.chat.completions.create(
            model='alibaba/tongyi-deepresearch-30b-a3b',
            messages=[{'role': 'user', 'content': 'Say hello'}],
            max_tokens=100
        )
        print('✅ SUCCESS!')
        print(f'Response: {response.choices[0].message.content}')
    except Exception as e:
        print(f'❌ FAILED: {str(e)}')

# 2. Serper API
print('\n2. SERPER API TEST')
print('-'*70)
serper_key = os.getenv('SERPER_KEY_ID')
print(f'API Key found: {"Yes" if serper_key else "No"}')
if serper_key:
    print(f'API Key (first 15 chars): {serper_key[:15]}...')
    try:
        response = requests.post(
            'https://google.serper.dev/search',
            headers={'X-API-KEY': serper_key, 'Content-Type': 'application/json'},
            json={'q': 'test', 'num': 1},
            timeout=10
        )
        print(f'Status Code: {response.status_code}')
        if response.status_code == 200:
            print('✅ SUCCESS!')
            data = response.json()
            print(f'Results: {len(data.get("organic", []))} items')
        else:
            print(f'❌ FAILED')
            print(f'Response: {response.text}')
    except Exception as e:
        print(f'❌ EXCEPTION: {str(e)}')

# 3. Jina API
print('\n3. JINA API TEST')
print('-'*70)
jina_keys = os.getenv('JINA_API_KEYS')
print(f'API Key found: {"Yes" if jina_keys else "No"}')
if jina_keys:
    jina_key = jina_keys.split(',')[0] if ',' in jina_keys else jina_keys
    print(f'API Key (first 15 chars): {jina_key[:15]}...')
    try:
        response = requests.get(
            'https://r.jina.ai/https://example.com',
            headers={'Authorization': f'Bearer {jina_key}'},
            timeout=10
        )
        print(f'Status Code: {response.status_code}')
        if response.status_code == 200:
            print('✅ SUCCESS!')
            print(f'Content length: {len(response.text)} chars')
        else:
            print(f'❌ FAILED')
            print(f'Response: {response.text[:300]}')
    except Exception as e:
        print(f'❌ EXCEPTION: {str(e)}')

# 4. Judge Model API
print('\n4. JUDGE MODEL API TEST (for evaluation)')
print('-'*70)
judge_key = os.getenv('API_KEY')
judge_base = os.getenv('BASE_URL')
print(f'API Key found: {"Yes" if judge_key else "No"}')
print(f'Base URL: {judge_base if judge_base else "Not set"}')
if judge_key and judge_base:
    print(f'API Key (first 15 chars): {judge_key[:15]}...')
    try:
        client = OpenAI(
            base_url=judge_base,
            api_key=judge_key
        )
        response = client.chat.completions.create(
            model='openai/o3-mini',
            messages=[{'role': 'user', 'content': 'Say hello'}],
            max_tokens=50
        )
        print('✅ SUCCESS!')
        print(f'Response: {response.choices[0].message.content}')
    except Exception as e:
        print(f'❌ FAILED: {str(e)}')

print('\n' + '='*70)
print('END OF API TESTING')
print('='*70)