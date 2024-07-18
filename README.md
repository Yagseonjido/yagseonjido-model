# 🚀 Model Server Docker Setup Guide

Follow these steps to set up and run your model server using Docker. Let's get started! 🛠️

## Prerequisites

Before you begin, make sure you have the following installed:

-   🐳 Docker Desktop (running)
-   🐙 Git

## Steps

1.  **📂 Clone the Repository**

    Open your terminal and run the following command to clone the repository:

    ```bash
    git clone https://github.com/Yagseonjido/yagseonjido-model.git
    ```

2.  **🚀 Navigate to the Project Directory**

    Change your current directory to the project folder:

    ```bash
    cd yagseonjido-model
    ```

3.  **🔧 Prepare the Environment File**

    Create a .env file inside the APP folder. The .env file should contain all the necessary environment variables required for your application. For example:

    ```bash
    OPENAI_API_KEY='OpenAI-API-Key'
    ```

4.  **🏗️ Build the Docker Image**

    Ensure Docker Desktop is running, then build the Docker image using the following command:

    ```bash
    docker build -t model-server .
    ```

5.  **🚢 Run the Docker Container**

    After the image is built, run the Docker container using:

    ```bash
    docker run -d -p 8000:8000 model-server
    ```

6.  **🔍 Test the POST API**

    Once the container is running, you can test the POST API using curl or any API client like Postman. For example, using curl:

    ```bash
    curl -X POST "http://localhost:8000/simulation/result/" -H "Content-Type: application/json" -d '{
        "name": "홍길동",
        "birthdate": "990818",
        "sex": "male",
        "weight": 70,
        "height": 175,
        "bloodPressure": "120/80",
        "pastDiseases": "고혈압, 당뇨",
        "currentMedications": "아스피린 100mg, 하루 1회",
        "allergies": "페니실린",
        "familyHistory": "고혈압, 암",
        "symptoms": "두통, 복통",
        "onset": "지난주",
        "painLevel": 10
    }'

    ```

    ```bash
    curl -X POST "http://localhost:8000/simulation/patient/" -H "Content-Type: application/json" -d '{
        "name": "홍길동",
        "birthdate": "1999-08-18",
        "sex": "male",
        "weight": 70,
        "height": 175,
        "bloodPressure": "120/80",
        "pastDiseases": "고혈압, 당뇨",
        "currentMedications": "아스피린 100mg, 하루 1회",
        "allergies": "페니실린",
        "familyHistory": "고혈압, 암",
        "symptoms": "두통, 복통",
        "onset": "지난주",
        "painLevel": 10,
        "explainType": 1,
        "scenario": {
            "prescription1": [
            {
                "drugId": "N02BE01",
                "drugType": "아세트 아미노펜",
                "labelKo": "타이레놀",
                "labelEn": "Tylenol",
                "efficacy": ["통증 완화", "해열"]
            }
            ],
            "reaction1": [
            {
                "label": "간 기능 저하",
                "explain": {
                "발생 빈도": "매우 드물(<0.1%)",
                "증상 예시": "황달, 어두운 소변"
                },
                "alert": "간 기능 저하 환자 주의"
            }
            ],
            "prescription2": [
            {
                "drugId": "A05BA03",
                "drugType": "간 보호제",
                "labelKo": "실리마린",
                "labelEn": "Silymarin",
                "efficacy": ["간 보호", "항산화 작용"]
            }
            ],
            "reaction2": [
            {
                "label": "위장관 문제",
                "explain": {
                "발생 빈도": "혼합(1-10%)",
                "증상 예시": "복통, 속쓰림, 구토"
                },
                "alert": "음식과 함께 복용"
            }
            ]
        }
    }'

    ```
