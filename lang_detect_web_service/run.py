# 서비스 메인코드 작성(엔트리(진입로) 코드) -> numpy
# 1. 모듈 가져오기
from flask import Flask, request, render_template, jsonify
# 남이 만든 모듈을 내가 가져다가 사용
from model import predictMainStream

# 2. 플라스크 객체 생성
app = Flask(__name__)
'''
    - __name__을 기술한 py에서 직접 실행되면 __main__ 으로 리턴
    - 다른 모듈에서 이 py를 호출, 모듈가져오기 등등 사용하면 파일명으로 리턴 : 조연
'''

# 3. 라우팅 (서버측으로 요청이 들어오면(주소/url)을 분석해서 어떤 함수가 대응할지 처리하는 것)
@app.route('/') # 홈페이지 url
def home():
    # 텍스트를 입력받는 화면 준비해서 클라이언트한테 랜더링하여 제공(서버 사이드 렌더링: SSR)
    # 응답 내용 -> 이 주소를 요청한 클라이언트가 보는 화면의 재료
    return render_template('index.html') # 홈페이지 열면 hi 함

# POST 방식으로 url이 반응하게 허용해야 함. 기본은 GET 방식
@app.route('/predict', methods=['POST']) # 나열 
def predict():
    # json으로 응답, 예측행위는 생략(임시)
    # 1. 클라이언트가 보낸 데이터 획득(post 방식)
    law_text = request.form.get('key')
    print(law_text) # 언어감지 클릭시 내용 불러옴
    # 2. 전처리(훈련중일 때는 정규식 사용이 이미 답을 알고 있어서 혼선이 없었음, 예측시에는 문제됨)
    #    입력 데이터의 최종 형태 -> (1, 65536), DataFrame 형태로 입력
    # 3. 모델에 입력 후 예측 수행
    y_pred = predictMainStream( law_text )
    # 답안지, 결과를 원하는 형태로 변형
    key = y_pred[0]
    dic = {
        'en':'영어',
        'fr':'프랑스어',
        'tl':'타갈리아어',
        'id':'인도네시아어',
        'ko':'한국어',
        'jp':'일본어',
    }

    # 4. 예측 결과를 받아서 응답처리
    return jsonify({'code':1, 'value':dic[ key ]})

# 4. 서버가동
if __name__ == '__main__': # 이 코드가 엔트리 포인트가 되면 서버가동
    # debug=True: 코드를 수정하면 실시간으로 리로드되어 반영됨
    app.run(debug=True)