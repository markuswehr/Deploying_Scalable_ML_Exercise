import requests

input_data = {
    "age": 30,
    "workclass": "Private",
    "fnlgt": 215646,
    "education": "Master",
    "education-num": 18,
    "marital-status": "Never-married",
    "occupation": "Tech-suppor",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "Germany"
}

response = requests.post(
    "https://deploying-scalable-ml-exercise.herokuapp.com/model_inference",
    json=input_data)
print(f'Response status code: {response.status_code}')
print(f'Response body: {response.json()}')