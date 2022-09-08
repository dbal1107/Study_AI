from flask import Flask, request, render_template, jsonify
import nlp

app = Flask(__name__)

@app.route('/')
def home():
    # SSR 처리시 데이터를 전달할 수 있다.
    return render_template('index.html', title='SBert 임베딩 기술을 활용한 챗봇')

@app.route('/query', methods=['POST'] ) #404:쿼리넣기 405 POST 넣기
def query():
    # TODO S1. 사용자가 보낸 메세지 추출
    # TODO S2. 미리 준비된 모델의 인코더를 통해 메세지를 임베딩한다.
    # TODO S3. 유사도 검사를 통해 가장 가까운 답변을 획득한다.
    # TODO S4. 응답
    res = {
        'code':1,
        'name':'고객센터',
        'msg':nlp.check_answer_similar(request.form.get('msg'))
    }
    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True)
