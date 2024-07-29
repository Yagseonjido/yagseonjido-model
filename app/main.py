# app/main.py
from fastapi import FastAPI, HTTPException
from openai import OpenAI
import os
from pydantic import BaseModel
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Get your OpenAI API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

class Prescription(BaseModel):
    drugId: str
    drugType: str
    labelKo: str
    labelEn: str
    efficacy: List[str]

class Reaction(BaseModel):
    label: str
    explain: Dict[str, str]
    alert: str

class PatientScenario(BaseModel):
    prescription1: List[Prescription]
    reaction1: List[Reaction]
    prescription2: List[Prescription]
    reaction2: List[Reaction]

class PatientInfo(BaseModel):
    name: str
    birthdate: str
    sex: str
    weight: int
    height: int
    bloodPressure: str
    pastDiseases: str
    currentMedications: str
    allergies: str
    familyHistory: str
    symptoms: str
    onset: str
    painLevel: int

class SimulateResult(BaseModel):
    name: str
    birthdate: str
    sex: str
    weight: int
    height: int
    bloodPressure: str
    pastDiseases: str
    currentMedications: str
    allergies: str
    familyHistory: str
    symptoms: str
    onset: str
    painLevel: int
    explainType: int
    scenario: PatientScenario

@app.post("/simulation/result/")
async def simulate_result(patient_info: PatientInfo):
    # Construct the prompt with example input and output
    prompt = f"""
    너는 의사를 도와주는 어시스턴트야. 입력받은 환자 정보를 바탕으로 환자에게 적절한 처방약들, 처방약들로 인해 환자에게 발생할 수 있는 각각의 부작용, 그리고 그 부작용이 초래할 수 있는 연쇄 처방 약물들과 그 약물들로 인한 부작용들을 노드와 엣지 형태로 제공하고 이 모든 결과들에 대해서 종합적 판단과 이유를 생성해.
    예시는 다음과 같고, painLevel은 1부터 10까지의 정수로 표현되며, 1이 가장 낮은 수준의 통증을 의미하고 10이 가장 높은 수준의 통증을 의미해.
    예시 1 입력:
    {{
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
    }}

    예시 1 출력:
    {{
        "nodes": [
        {{
            "id": "prescription1-1",
            "drugId": "N02BE01",
            "drugType": "아세트 아미노펜"
            "labelKo": "타이레놀",
            "labelEn": "Tylenol",
            "efficacy": ["통증 완화", "해열"]
        }},
        {{
            "id": "prescription1-2",
            "drugId": "M01AE01",
            "drugType": "NSAIDs"
            "labelKo": "이부프로펜",
            "labelEn": "Ibuprofen",
            "efficacy": ["통증 완화", "염증 감소"]
        }},
        {{
            "id": "prescription1-3",
            "drugId": "M01AE01",
            "drugType": "NSAIDs"
            "labelKo": "나프록센",
            "labelEn": "Naproxen",
            "efficacy": ["통증 완화", "염증 감소"]
        }},  
        {{
			"id": "reaction1-1",
			"label":"간 기능 저하",
			"explain": 
			{{
					"발생 빈도" : "매우 드물(<0.1%)",
					"증상 예시" : "황달, 어두운 소변"
			}}
			"alert" : "간 기능 저하 환자 주의"
		}},
		{{
			"id": "reaction1-2",
			"label":"위장관 문제",
			"explain": 
			{{
					"발생 빈도" : "혼합(1-10%)",
					"증상 예시" : "복통, 속쓰림, 구토"
			}}
			"alert" : "음식과 함께 복용"
		}},
		{{
			"id": "reaction1-3",
			"label":"신장 기능 저하",
			"explain": 
			{{
					"발생 빈도" : "드물(0.1-1%)",
					"증상 예시" : "소변량 감소, 피로, 부종"
			}}
			"alert" : "신장 모니터링"
		}},
    {{
      "id": "prescription2-1",
      "drugId": "A05BA03",
      "drugType": "간 보호제"
      "labelKo": "실리마린",
      "labelEn": "Silymarin",
      "efficacy": ["간 보호", "항산화 작용"]
    }},
    {{
      "id": "prescription2-2",
      "drugId": "A02BC01",
      "drugType": "PPI"
      "labelKo": "오메프라졸",
      "labelEn": "Prilosec",
      "efficacy": ["위장관 장애"]
    }},
    {{
      "id": "prescription2-3",
      "drugId": "A02BA02",
      "drugType": "H2RA"
      "labelKo": "라니티딘",
      "labelEn": "Ranitidine",
      "efficacy": ["위장 분비 감소", "위장 보호"]
    }},  
    {{
      "id": "prescription2-4",
      "drugId": "C03CA01",
      "drugType": "이뇨제"
      "labelKo": "푸로세이드",
      "labelEn": "Furosemide",
      "efficacy": ["체액 배출", "부종 감소"]
    }},  
    {{
      "id": "prescription2-5",
      "drugId": "C09AA03",
      "drugType": "신장 보호 약물"
      "labelKo": "리시노프릴",
      "labelEn": "Lisinopril",
      "efficacy": ["신장 보호", "혈압 조절"]
    }},  
    {{
			"id": "reaction2-1",
			"label":"전해질 불균형",
			"explain": 
			{{
					"발생 빈도" : "혼합(1-10%)",
					"증상 예시" : "저칼륨혈증, 저마그네슘혈증"
			}}
			"alert" : "간 기능 저하 환자 주의"
		}},
		{{
			"id": "reaction2-2",
			"label":"두통",
			"explain": 
			{{
					"발생 빈도" : "혼합(1-10%)",
					"증상 예시" : "경미한 두통"
			}}
			"alert" : "필요 시 진통제 복용"
		}},
		{{
			"id": "reaction2-3",
			"label":"위장관 문제",
			"explain": 
			{{
					"발생 빈도" : "혼합(1-10%)",
					"증상 예시" : "복통, 속쓰림, 구토"
			}}
			"alert" : "음식과 함께 복용"
		}},
		{{
			"id": "reaction2-4",
			"label":"탈수 및 저혈압",
			"explain": 
			{{
					"발생 빈도" : "드물(0.1%-1%)",
					"증상 예시" : "어지러움, 피로, 저혈압"
			}}
			"alert" : "체액 상태 모니터링 필요"
		}},
		{{
			"id": "reaction2-5",
			"label": "통풍",
			"explain": 
			{{
					"발생 빈도" : "드물(0.1%-1%)",
					"증상 예시" : "어지러움, 피로, 저혈압"
			}}
			"alert" : "체액 상태 모니터링 필요"
		}},
  ],
  "edges": [
    {{
      "from": "prescription1-1",
      "to": "reaction1-1"
    }},
    {{
      "from": "prescription1-2",
      "to": "reaction1-2"
    }},
    {{
      "from": "prescription1-2",
      "to": "reaction1-3"
    }},
    {{
      "from": "prescription1-3",
      "to": "reaction1-2"
    }},
    {{
      "from": "prescription1-3",
      "to": "reaction1-3"
    }},
    {{
      "from": "reaction1-1",
      "to": "prescription2-1"
    }},
    {{
      "from": "reaction1-2",
      "to": "prescription2-2"
    }},
    {{
      "from": "reaction1-2",
      "to": "prescription2-3"
    }},
    {{
      "from": "reaction1-3",
      "to": "prescription2-4"
    }},
    {{
      "from": "reaction1-3",
      "to": "prescription2-5"
    }},
    {{
      "from": "prescription2-1",
      "to": "reaction2-3"
    }},
    {{
      "from": "prescription2-2",
      "to": "reaction2-1"
    }},
    {{
      "from": "prescription2-3",
      "to": "reaction2-2"
    }},
    {{
      "from": "prescription2-4",
      "to": "reaction2-2"
    }},
    {{
      "from": "prescription2-4",
      "to": "reaction2-4"
    }},     
	  {{
      "from": "prescription2-4",
      "to": "reaction2-5"
    }},
    {{
      "from": "prescription2-5",
      "to": "reaction2-4"
    }}
  ],
  "review" : "홍길동 환자는 두통과 복통으로 인해 타이레놀(아세트아미노펜)과 이부프로펜(NSAIDs) 또는 나프록센(NSAIDs)이 처방될 수 있습니다. 그러나 이 약물들은 간 기능 저하 및 위장관 문제, 신장 기능 저하 등의 부작용을 초래할 수 있으므로 주의가 필요합니다. 특히, 환자는 이미 고혈압과 당뇨 병력을 가지고 있으며, 아스피린을 복용 중이기 때문에 NSAIDs의 사용은 신중하게 고려되어야 합니다. 간 보호를 위해 실리마린이 처방될 수 있으며, 위장관 문제를 예방하기 위해 오메프라졸이나 라니티딘이 추가적으로 처방될 수 있습니다. 추가적으로, 푸로세이드와 리시노프릴을 통해 신장 보호와 혈압 조절을 보완할 수 있으나, 이 경우 탈수 및 저혈압 등의 부작용을 모니터링해야 합니다. 따라서, 홍길동 환자는 위장관 장애가 있고, 간 및 신장 기능 저하를 주의하면서 복용해야 하며, 필요한 경우 진통제를 복용할 수 있습니다. 약물 복용 시 음식과 함께 복용하여 위장관 문제를 최소화하고, 정기적인 간 및 신장 모니터링이 필요합니다."
}}

예시 2 입력:
{{
	"name": "최수진",
	"birthdate": "900812",
	"sex": "female",
	"weight": 55,
	"height": 160,
	"bloodPressure": "120/80",
	"pastDiseases": "당뇨, 편두통",
	"currentMedications": "메트포르민 500mg, 하루 2회",
	"allergies": "없음",
	"familyHistory": "당뇨, 고혈압",
	"symptoms": "두통, 어지러움",
	"onset": "3일 전",
	"painLevel": 8
}}

예시 2 출력:
{{
  "nodes": [
    {{
      "id": "prescription1-1",
      "drugId": "N02CC01",
      "drugType": "트립탄",
      "labelKo": "수마트립탄",
      "labelEn": "Sumatriptan",
      "efficacy": ["편두통 완화"]
    }},
    {{
      "id": "prescription1-2",
      "drugId": "N02BE01",
      "drugType": "아세트 아미노펜",
      "labelKo": "타이레놀",
      "labelEn": "Tylenol",
      "efficacy": ["통증 완화", "해열"]
    }},
    {{
      "id": "reaction1-1",
      "label": "심계항진",
      "explain": {{
        "발생 빈도": "드물(0.1-1%)",
        "증상 예시": "빠른 심장박동, 두근거림"
      }},
      "alert": "심장 질환자 주의"
    }},
    {{
      "id": "reaction1-2",
      "label": "위장관 문제",
      "explain": {{
        "발생 빈도": "혼합(1-10%)",
        "증상 예시": "복통, 속쓰림, 구토"
      }},
      "alert": "음식과 함께 복용"
    }},
    {{
      "id": "prescription2-1",
      "drugId": "C07AB03",
      "drugType": "베타차단제",
      "labelKo": "아테놀롤",
      "labelEn": "Atenolol",
      "efficacy": ["심박수 조절", "고혈압 조절"]
    }},
    {{
      "id": "prescription2-2",
      "drugId": "A02BC01",
      "drugType": "PPI",
      "labelKo": "오메프라졸",
      "labelEn": "Prilosec",
      "efficacy": ["위장관 장애"]
    }},
    {{
      "id": "reaction2-1",
      "label": "저혈압",
      "explain": {{
        "발생 빈도": "혼합(1-10%)",
        "증상 예시": "어지러움, 피로, 저혈압"
      }},
      "alert": "저혈압 증상 모니터링 필요"
    }},
    {{
      "id": "reaction2-2",
      "label": "전해질 불균형",
      "explain": {{
        "발생 빈도": "혼합(1-10%)",
        "증상 예시": "저칼륨혈증, 저마그네슘혈증"
      }},
      "alert": "전해질 수치 모니터링 필요"
    }}
  ],
  "edges": [
    {{
      "from": "prescription1-1",
      "to": "reaction1-1"
    }},
    {{
      "from": "prescription1-2",
      "to": "reaction1-2"
    }},
    {{
      "from": "reaction1-1",
      "to": "prescription2-1"
    }},
    {{
      "from": "reaction1-2",
      "to": "prescription2-2"
    }},
    {{
      "from": "prescription2-1",
      "to": "reaction2-1"
    }},
    {{
      "from": "prescription2-2",
      "to": "reaction2-2"
    }}
  ],
  "review": "최수진 환자는 두통과 어지러움으로 인해 수마트립탄(트립탄)과 타이레놀(아세트아미노펜)이 처방될 수 있습니다. 수마트립탄은 편두통 완화에 효과적이지만, 심계항진과 위장관 문제 등의 부작용이 있을 수 있습니다. 심계항진에 대해 아테놀롤(베타차단제)을 처방할 수 있으며, 이 경우 저혈압 등의 부작용을 모니터링해야 합니다. 위장관 문제를 예방하기 위해 오메프라졸(PPI)이 추가적으로 처방될 수 있으며, 전해질 불균형 등의 부작용을 모니터링해야 합니다. 환자는 당뇨와 편두통 병력이 있으므로 심장 질환 주의가 필요하며, 필요 시 진통제를 복용할 수 있습니다. 정기적인 건강 검진과 휴식, 적절한 수면이 필요합니다."
}}

예시 3 입력:
{{
  "name": "이민호",
  "birthdate": "850305",
  "sex": "male",
  "weight": 85,
  "height": 178,
  "bloodPressure": "140/90",
  "pastDiseases": "고혈압, 신장질환",
  "currentMedications": "베타차단제 50mg, 하루 1회",
  "allergies": "없음",
  "familyHistory": "고혈압, 신장질환",
  "symptoms": "피로, 복통",
  "onset": "3일 전",
  "painLevel": 6
}}

예시 3 출력:
{{
  "nodes": [
    {{
      "id": "prescription1-1",
      "drugId": "N02AX02",
      "drugType": "비마약성 진통제",
      "labelKo": "트라마돌",
      "labelEn": "Tramadol",
      "efficacy": ["통증 완화"]
    }},
    {{
      "id": "prescription1-2",
      "drugId": "A03FA01",
      "drugType": "항경련제",
      "labelKo": "가바펜틴",
      "labelEn": "Gabapentin",
      "efficacy": ["신경통 완화"]
    }},
    {{
      "id": "prescription1-3",
      "drugId": "A02BA02",
      "drugType": "H2RA",
      "labelKo": "라니티딘",
      "labelEn": "Ranitidine",
      "efficacy": ["위산 분비 감소", "위 보호"]
    }},
    {{
      "id": "reaction1-1",
      "label": "졸음",
      "explain": {{
        "발생 빈도": "일반적(1-10%)",
        "증상 예시": "졸음, 피로"
      }},
      "alert": "운전 및 기계 조작 주의"
    }},
    {{
      "id": "reaction1-2",
      "label": "어지러움",
      "explain": {{
        "발생 빈도": "일반적(1-10%)",
        "증상 예시": "현기증, 균형감 상실"
      }},
      "alert": "갑작스런 자세 변화 주의"
    }},
    {{
      "id": "reaction1-3",
      "label": "위장관 문제",
      "explain": {{
        "발생 빈도": "혼합(1-10%)",
        "증상 예시": "복통, 구토"
      }},
      "alert": "음식과 함께 복용"
    }},
    {{
      "id": "prescription2-1",
      "drugId": "A10BB12",
      "drugType": "DPP-4 억제제",
      "labelKo": "시타글립틴",
      "labelEn": "Sitagliptin",
      "efficacy": ["혈당 조절"]
    }},
    {{
      "id": "prescription2-2",
      "drugId": "B01AC06",
      "drugType": "항응고제",
      "labelKo": "아스피린",
      "labelEn": "Aspirin",
      "efficacy": ["혈액 희석"]
    }},
    {{
      "id": "prescription2-3",
      "drugId": "C09AA05",
      "drugType": "ACE 억제제",
      "labelKo": "에나프릴",
      "labelEn": "Enalapril",
      "efficacy": ["혈압 조절"]
    }},
    {{
      "id": "reaction2-1",
      "label": "저혈압",
      "explain": {{
        "발생 빈도": "혼합(1-10%)",
        "증상 예시": "어지러움, 피로"
      }},
      "alert": "혈압 모니터링 필요"
    }},
    {{
      "id": "reaction2-2",
      "label": "위장관 문제",
      "explain": {{
        "발생 빈도": "혼합(1-10%)",
        "증상 예시": "복통, 속쓰림, 구토"
      }},
      "alert": "음식과 함께 복용"
    }},
    {{
      "id": "reaction2-3",
      "label": "두통",
      "explain": {{
        "발생 빈도": "혼합(1-10%)",
        "증상 예시": "경미한 두통"
      }},
      "alert": "필요 시 진통제 복용"
    }},
    {{
      "id": "reaction2-4",
      "label": "저혈당",
      "explain": {{
        "발생 빈도": "드물(0.1%-1%)",
        "증상 예시": "현기증, 피로, 저혈당"
      }},
      "alert": "혈당 상태 모니터링 필요"
    }}
  ],
  "edges": [
    {{
      "from": "prescription1-1",
      "to": "reaction1-1"
    }},
    {{
      "from": "prescription1-1",
      "to": "reaction1-2"
    }},
    {{
      "from": "prescription1-2",
      "to": "reaction1-3"
    }},
    {{
      "from": "prescription1-3",
      "to": "reaction1-2"
    }},
    {{
      "from": "reaction1-1",
      "to": "prescription2-1"
    }},
    {{
      "from": "reaction1-2",
      "to": "prescription2-2"
    }},
    {{
      "from": "reaction1-3",
      "to": "prescription2-3"
    }},
    {{
      "from": "prescription2-1",
      "to": "reaction2-4"
    }},
    {{
      "from": "prescription2-2",
      "to": "reaction2-1"
    }},
    {{
      "from": "prescription2-3",
      "to": "reaction2-2"
    }},
    {{
      "from": "prescription2-3",
      "to": "reaction2-3"
    }}
  ],
  "review": "이민호 환자는 피로와 복통으로 인해 트라마돌(비마약성 진통제)와 가바펜틴(항경련제) 또는 라니티딘(H2RA)이 처방될 수 있습니다. 그러나 이 약물들은 졸음, 어지러움, 위장관 문제 등의 부작용을 초래할 수 있으므로 주의가 필요합니다. 특히, 환자는 고혈압과 신장질환 병력을 가지고 있으며, 베타차단제를 복용 중이기 때문에 항경련제의 사용은 신중하게 고려되어야 합니다. 위 보호를 위해 라니티딘이 처방될 수 있으며, 혈당 조절을 위해 시타글립틴이 추가적으로 처방될 수 있습니다. 추가적으로, 에나프릴을 통해 혈압 조절을 보완할 수 있으나, 이 경우 저혈압 및 저혈당 등의 부작용을 모니터링해야 합니다. 따라서, 이민호 환자는 위장관 장애가 있고, 전해질 및 혈압 상태를 주의하면서 복용해야 하며, 필요한 경우 진통제를 복용할 수 있습니다. 약물 복용 시 음식과 함께 복용하여 위장관 문제를 최소화하고, 정기적인 혈압 및 혈당 모니터링이 필요합니다."
}}

입력 예시:
    {{
        "name": "{patient_info.name}",
        "birthdate": "{patient_info.birthdate}",
        "sex": "{patient_info.sex}",
        "weight": {patient_info.weight},
        "height": {patient_info.height},
        "bloodPressure": "{patient_info.bloodPressure}",
        "pastDiseases": "{patient_info.pastDiseases}",
        "currentMedications": "{patient_info.currentMedications}",
        "allergies": "{patient_info.allergies}",
        "familyHistory": "{patient_info.familyHistory}",
        "symptoms": "{patient_info.symptoms}",
        "onset": "{patient_info.onset}",
        "painLevel": {patient_info.painLevel}
    }}

    예시 1, 2, 3을 참조하여 위의 입력 데이터를 기반으로, 위 출력 형식과 유사한 형식의 출력 데이터를 생성해.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 의사를 도와주는 어시스턴트입니다. 모든 응답은 JSON 형식으로 제공해야 합니다."},
            {"role": "user", "content": prompt}
        ],
    )

    return {response.choices[0].message.content}

@app.post("/simulation/patient/")
async def simulate_patient(patient_info: SimulateResult):
    explain_prompts = {
        1: "학생들도 이해할 수 있도록 쉽고 친절하게 설명해.",
        2: "간단하고 핵심적인 정보만 제공해.",
        3: "전문의나 약사가 설명하는 수준으로 깊이 있게, 2줄 이상으로 자세하게 설명을 생성해."
    }

    scenario = patient_info.scenario
    # scenario_text = ""
    
    # for prescription in scenario.prescription1:
    #     scenario_text += f"처방 약물: {prescription.labelKo}({prescription.labelEn}) - {prescription.drugType}\n"
    #     scenario_text += f"효능: {', '.join(prescription.efficacy)}\n"
    
    # for reaction in scenario.reaction1:
    #     scenario_text += f"부작용: {reaction.label}\n"
    #     scenario_text += f"설명: {reaction.explain['발생 빈도']} - {reaction.explain['증상 예시']}\n"
    #     scenario_text += f"주의사항: {reaction.alert}\n"

    # for prescription in scenario.prescription2:
    #     scenario_text += f"추가 처방 약물: {prescription.labelKo}({prescription.labelEn}) - {prescription.drugType}\n"
    #     scenario_text += f"효능: {', '.join(prescription.efficacy)}\n"

    # for reaction in scenario.reaction2:
    #     scenario_text += f"추가 부작용: {reaction.label}\n"
    #     scenario_text += f"설명: {reaction.explain['발생 빈도']} - {reaction.explain['증상 예시']}\n"
    #     scenario_text += f"주의사항: {reaction.alert}\n"
    
    # prescription1_text = ''.join(map(lambda p: f"처방 약물: {p.labelKo}({p.labelEn}) - {p.drugType}\n효능: {', '.join(p.efficacy)}\n", scenario.prescription1))
    # reaction1_text = ''.join(map(lambda r: f"부작용: {r.label}\n설명: {r.explain['발생 빈도']} - {r.explain['증상 예시']}\n주의사항: {r.alert}\n", scenario.reaction1))
    # prescription2_text = ''.join(map(lambda p: f"추가 처방 약물: {p.labelKo}({p.labelEn}) - {p.drugType}\n효능: {', '.join(p.efficacy)}\n", scenario.prescription2))
    # reaction2_text = ''.join(map(lambda r: f"추가 부작용: {r.label}\n설명: {r.explain['발생 빈도']} - {r.explain['증상 예시']}\n주의사항: {r.alert}\n", scenario.reaction2))

    # scenario_text = f"{prescription1_text}{reaction1_text}{prescription2_text}{reaction2_text}"

    scenario_text = f"""
    처방 약물 1:
    {scenario.prescription1}
    
    부작용 1:
    {scenario.reaction1}
    
    추가 처방 약물:
    {scenario.prescription2}
    
    추가 부작용:
    {scenario.reaction2}
    """

    # Example
    example = """
    예시1 입력:
    {{
        "name": "홍길동",
        "birthdate": "1999-08-18"
        "sex": "male"
        "weight": 70,
        "height": 175,
        "bloodPressure": "120/80"
        "pastDiseases": "고혈압, 당뇨",
        "currentMedications": "아스피린 100mg, 하루 1회",
        "allergies": "페니실린",
        "familyHistory": "고혈압, 암",
        "symptoms": "두통, 복통",
        "onset": "지난주"
        "painLevel": 10,
        "explainType": 1,
        "scenario": {{
        "prescription1": [
            {{
            "drugId": "N02BE01",
            "drugType": "아세트 아미노펜"
            "labelKo": "타이레놀",
            "labelEn": "Tylenol",
            "efficacy": ["통증 완화", "해열"]
            }}
        ],
        "reaction1": [
            {{
                    "label":"간 기능 저하",
                    "explain": 
                    {{
                            "발생 빈도" : "매우 드물(<0.1%)",
                            "증상 예시" : "황달, 어두운 소변"
                    }}
                    "alert" : "간 기능 저하 환자 주의"
                }}
        ],
        "prescription2": [
            {{
            "drugId": "A05BA03",
            "drugType": "간 보호제"
            "labelKo": "실리마린",
            "labelEn": "Silymarin",
            "efficacy": ["간 보호", "항산화 작용"]
            }}
        ],
        "reaction2": [
            {{
                    "label":"위장관 문제",
                    "explain": 
                    {{
                            "발생 빈도" : "혼합(1-10%)",
                            "증상 예시" : "복통, 속쓰림, 구토"
                    }}
                    "alert" : "음식과 함께 복용"
                }}
        ]
        }}
    }}
예시1 출력:
    {{
      "prescription1": [
        {{
            "drugName": "타이레놀",
            "explain": "타이레놀은 두통과 발열을 완화하는 데 도움을 주는 약입니다. 타이레놀은 아세트아미노펜 성분을 포함하고 있으며, 이 성분은 통증을 줄이고 열을 내리는 데 탁월합니다. 현재 두통과 복통으로 고생하고 계시기 때문에 타이레놀이 증상을 완화하는 데 효과적일 것입니다. 하지만 매일 100mg의 아스피린을 복용 중이시므로 타이레놀을 과다 복용하지 않도록 주의해주세요.",
            "effects": [
                {{"통증 완화": "타이레놀은 아픈 느낌을 줄여줍니다. 특히 두통과 같은 일상적인 통증을 완화하는 데 효과적입니다."}},
                {{"해열 작용": "타이레놀은 열을 낮추어 체온을 정상으로 유지합니다. 발열로 인한 불편함을 줄여줍니다."}}
            ]
        }}
      ],
      "reaction1": [
        {{
            "label": "간 기능 저하",
            "symptom": [
                {{"황달": "피부나 눈이 노래질 수 있습니다. 이는 간 기능이 저하되어 빌리루빈이 축적되기 때문입니다."}},
                {{"어두운 소변": "소변 색이 어두워질 수 있습니다. 이는 간 기능이 저하되어 빌리루빈이 소변으로 배출되기 때문입니다."}}
            ],
            "explain": "홍길동님, 타이레놀을 많이 복용하면 간이 손상될 수 있어요. 과거에 고혈압과 당뇨를 앓으셨기 때문에 간 건강에 특히 주의가 필요합니다. 간 기능이 저하되면 황달이나 어두운 소변과 같은 증상이 나타날 수 있습니다.",
            "goodHabit": [
                {{"복용약 준수": "의사나 약사가 정해준 양을 꼭 지켜서 복용해야 해요. 복용량을 준수하면 부작용을 줄일 수 있습니다."}}
            ],
            "badHabit": [
                {{"과다 복용 금지": "정해진 양보다 많이 복용하지 않도록 주의하세요. 과다 복용은 간 손상의 위험을 높입니다."}},
                {{"알코올 섭취 자제": "약을 복용하는 동안에는 술을 마시지 마세요. 알코올은 간에 추가적인 부담을 줄 수 있습니다."}}
            ]
        }}
      ],
      "prescription2": [
        {{
            "drugName": "실리마린",
            "explain": "실리마린은 간을 보호하고 건강하게 유지하는 데 도움을 주는 약입니다. 실리마린은 밀크씨슬에서 추출한 성분으로, 간 세포를 보호하고 재생을 촉진합니다. 과거에 고혈압과 당뇨를 앓으셨기 때문에 실리마린이 간 건강을 유지하는 데 도움이 될 것입니다.",
            "effects": [
                {{"간 보호": "실리마린은 간 세포의 재생을 돕고, 간 손상을 예방합니다. 특히, 간이 손상된 경우에도 간 기능을 회복시키는 데 도움을 줍니다."}},
                {{"항산화 작용": "실리마린은 항산화 작용을 통해 세포를 손상으로부터 보호합니다. 자유 라디칼을 제거하여 세포 손상을 방지합니다."}}
            ]
        }}
      ],
      "reaction2": [
        {{
            "label": "위장관 문제",
            "symptom": [
                {{"복통": "실리마린을 복용하면 위장관에 자극을 주어 배가 아플 수 있습니다."}},
                {{"속쓰림": "실리마린이 위산 분비를 자극하여 속이 쓰릴 수 있습니다."}}
            ],
            "explain": "홍길동님, 실리마린을 복용하면 배가 아프거나 속이 쓰릴 수 있어요. 이는 약물이 위장 점막을 자극하기 때문입니다. 음식을 먹고 약을 복용하면 이런 증상을 줄일 수 있습니다.",
            "goodHabit": [
                {{"음식과 함께 복용": "약을 음식과 함께 복용하면 위장관 문제가 덜 발생할 수 있습니다. 식사 후에 약을 복용하는 것이 좋습니다."}}
            ]
        }}
      ],
      "totalResult": [
        {{
          "explain": "결론적으로, 타이레놀과 실리마린을 복용할 때 간 건강과 위장 건강을 잘 관리해야 합니다. 두 약물 모두 특정 부작용이 있으므로 주의가 필요합니다. 타이레놀은 간 기능 저하의 위험이 있고, 실리마린은 위장관 문제를 일으킬 수 있습니다."
        }},
        {{
          "advice": [
            {{"간 건강 유지": "정기적으로 간 검사를 받아 간 건강을 확인하세요. 이는 간 손상을 조기에 발견하고 치료하는 데 중요합니다."}},
            {{"알코올 섭취 자제": "타이레놀을 복용하는 동안에는 알코올 섭취를 줄이세요. 알코올은 간 손상을 악화시킬 수 있습니다."}},
            {{"의사와 상담": "부작용이 발생하면 즉시 의사와 상담하세요. 의사가 필요한 조치를 제시해 줄 것입니다."}}
          ]
        }}
      ]
    }}

    예시 2 입력:
    {{
    "name": "최수진",
    "birthdate": "970402",
    "sex": "female",
    "weight": 55,
    "height": 160,
    "bloodPressure": "100/60",
    "pastDiseases": "천식, 편두통",
    "currentMedications": "살메테롤 25mcg, 하루 2회",
    "allergies": "없음",
    "familyHistory": "천식",
    "symptoms": "두통, 어지러움",
    "onset": "2일 전",
    "painLevel": 4
    "explainType": 2,
    "scenario": {{
	      "prescription1" : [{{
		      "drugId": "N02CC01",
		      "drugType": "트립탄",
		      "labelKo": "수마트립탄",
		      "labelEn": "Sumatriptan",
		      "efficacy": ["편두통 완화"]
		    }}],
		    "reaction1": [{{
		      "label": "심계항진",
		      "explain": {{
		        "발생 빈도": "드물(0.1-1%)",
		        "증상 예시": "빠른 심장박동, 두근거림"
		      }},
		      "alert": "심장 질환자 주의"
		    }}],
		    "prescription2" : [{{
		      "drugId": "C07AB03",
		      "drugType": "베타차단제",
		      "labelKo": "아테놀롤",
		      "labelEn": "Atenolol",
		      "efficacy": ["심박수 조절", "고혈압 조절"]
		    }}],
		    "reaction2": [{{
		      "label": "저혈압",
		      "explain": {
		        "발생 빈도": "혼합(1-10%)",
		        "증상 예시": "어지러움, 피로, 저혈압"
		      }},
		      "alert": "저혈압 증상 모니터링 필요"
		    }}]
    }}
}}

예시 2 출력:
{{
  "prescription1": [
    {{
        "drugName": "수마트립탄",
        "explain": "수마트립탄은 편두통을 완화하는 약입니다. 편두통이 발생했을 때 신속하게 통증을 줄여줍니다. 최수진님은 편두통으로 고생하고 계시기 때문에 수마트립탄이 도움될 것입니다. 하지만 심장에 부담을 줄 수 있으므로 복용 시 주의가 필요합니다.",
        "effects": [
            {{"편두통 완화": "수마트립탄은 편두통으로 인한 두통을 효과적으로 완화합니다."}}
        ]
    }}
  ],
  "reaction1": [
    {{
        "label": "심계항진",
        "symptom": [
            {{"빠른 심장박동": "수마트립탄을 복용하면 심장이 빠르게 뛸 수 있습니다."}},
            {{"두근거림": "가슴이 두근거리는 느낌이 들 수 있습니다."}}
        ],
        "explain": "최수진님, 수마트립탄을 복용하면 가끔 심장이 빠르게 뛰거나 두근거림을 느낄 수 있습니다. 이는 드문 부작용이지만, 심장에 문제가 있는 분들은 특히 주의가 필요합니다.",
        "goodHabit": [
            {{"복용약 준수": "의사의 지시에 따라 약을 정확히 복용해야 합니다."}}
        ],
        "badHabit": [
            {{"과다 복용 금지": "정해진 양을 넘지 않도록 주의하세요."}}
        ]
        }}
    ],
    "prescription2": [
        {{
            "drugName": "아테놀롤",
            "explain": "아테놀롤은 심박수와 혈압을 조절하는 약입니다. 심장이 너무 빠르게 뛰지 않도록 하고, 고혈압을 조절하는 데 도움을 줍니다.",
            "effects": [
                {{"심박수 조절": "아테놀롤은 심박수를 조절하여 심장 부담을 줄여줍니다."}},
                {{"고혈압 조절": "혈압을 낮추어 고혈압을 관리합니다."}}
            ]
        }}
    ],
    "reaction2": [
        {{
            "label": "저혈압",
            "symptom": [
                {{"어지러움": "혈압이 낮아지면 어지러울 수 있습니다."}},
                {{"피로": "저혈압으로 인해 피로감을 느낄 수 있습니다."}}
            ],
            "explain": "최수진님, 아테놀롤을 복용하면 혈압이 낮아질 수 있습니다. 혈압이 너무 낮아지면 어지러움이나 피로를 느낄 수 있으니 주의가 필요합니다.",
            "goodHabit": [
                {{"증상 모니터링": "저혈압 증상이 나타나면 즉시 의료진과 상담하세요."}}
            ]
        }}
    ],
    "totalResult": [
    {{
      "explain": "결론적으로, 수마트립탄과 아테놀롤을 복용할 때 심장 건강과 혈압을 잘 관리해야 합니다. 두 약물 모두 특정 부작용이 있으므로 주의가 필요합니다. 수마트립탄은 심계항진의 위험이 있고, 아테놀롤은 저혈압을 일으킬 수 있습니다."
    }},
    {{
      "advice": [
        {{"건강 모니터링": "정기적으로 건강 상태를 체크하고, 이상이 있으면 즉시 의사와 상담하세요."}},
        {{"의사와 상담": "부작용이 발생하면 즉시 의사와 상담하세요. 의사가 필요한 조치를 제시해 줄 것입니다."}}
      ]
    }}
  ]
}}

예시 3 입력:
{{
    "name": "이민호",
    "birthdate": "850305",
    "sex": "male",
    "weight": 85,
    "height": 178,
    "bloodPressure": "140/90",
    "pastDiseases": "고혈압, 신장질환",
    "currentMedications": "베타차단제 50mg, 하루 1회",
    "allergies": "없음",
    "familyHistory": "고혈압, 신장질환",
    "symptoms": "피로, 복통",
    "onset": "3일 전",
    "painLevel": 6,
    "explainType": 3,
    "scenario": {{
		    "prescription1" : [{{
		      "drugId": "N02AX02",
		      "drugType": "비마약성 진통제",
		      "labelKo": "트라마돌",
		      "labelEn": "Tramadol",
		      "efficacy": ["통증 완화"]
		    }}],
		    "reaction1": [{{
		      "label": "졸음",
		      "explain": {{
		        "발생 빈도": "일반적(1-10%)",
		        "증상 예시": "졸음, 피로"
		      }},
		      "alert": "운전 및 기계 조작 주의"
		    }}],
		    "prescription2" : [{{
		      "drugId": "A10BB12",
		      "drugType": "DPP-4 억제제",
		      "labelKo": "시타글립틴",
		      "labelEn": "Sitagliptin",
		      "efficacy": ["혈당 조절"]
		    }}],
		    "reaction2": [{{
		      "label": "저혈당",
		      "explain": {{
		        "발생 빈도": "드물(0.1%-1%)",
		        "증상 예시": "현기증, 피로, 저혈당"
		      }},
		      "alert": "혈당 상태 모니터링 필요"
		    }}]
    }}
}}

예시 3 출력:
{{
  "prescription1": [
    {{
        "drugName": "트라마돌",
        "explain": "이민호님, 트라마돌은 중등도에서 중증의 통증을 완화하는 비마약성 진통제입니다. 트라마돌은 중추신경계에 작용하여 통증 신호를 차단하고, 세로토닌과 노르에피네프린의 재흡수를 억제하여 통증 경감을 유도합니다. 이 약물은 복통과 피로로 인한 불편함을 경감시켜 줄 수 있습니다. 그러나 신장질환과 고혈압 병력이 있는 환자에게는 용량 조절과 주의 깊은 모니터링이 필요합니다. 트라마돌은 간에서 대사되며, 신장으로 배설되기 때문에 신장 기능이 저하된 경우 약물의 축적 위험이 있으므로 주의해야 합니다.",
        "effects": [
            {{"통증 완화": "트라마돌은 신경계에 작용하여 통증 신호를 차단하고, 중추신경계에서 세로토닌과 노르에피네프린의 재흡수를 억제하여 통증을 경감시킵니다. 이는 복통과 같은 중등도에서 중증의 통증을 효과적으로 완화합니다."}}
        ]
    }}
  ],
  "reaction1": [
    {{
        "label": "졸음",
        "symptom": [
            {{"졸음": "트라마돌을 복용하면 중추신경계에 영향을 미쳐 졸음이 올 수 있습니다."}},
            {{"피로": "약물의 신경계 작용으로 인해 피로감을 느낄 수 있습니다."}}
        ],
        "explain": "트라마돌은 중추신경계에 작용하여 졸음과 피로를 유발할 수 있습니다. 이는 신경계에서 세로토닌과 노르에피네프린의 재흡수를 억제하는 작용 메커니즘과 관련이 있습니다. 따라서 약물을 복용한 후에는 운전이나 기계 조작과 같은 집중력이 필요한 활동을 피하는 것이 중요합니다. 특히, 이민호님은 고혈압과 신장질환 병력이 있으므로 졸음이나 피로 증상이 나타날 경우 즉시 의료진과 상의해야 합니다. 트라마돌의 이러한 부작용은 특히 장기간 복용 시 더욱 주의가 필요합니다.",
        "goodHabit": [
            {{"운전 주의": "약물 복용 후 졸음이 올 수 있으므로 운전이나 기계 조작을 피하세요."}},
            {{"휴식": "피로감을 줄이기 위해 충분한 휴식을 취하세요."}}
        ],
        "badHabit": [
            {{"과다 복용 금지": "트라마돌을 과다 복용하지 않도록 주의하세요. 과다 복용은 부작용을 증가시킬 수 있습니다."}}
        ]
    }}
  ],
  "prescription2": [
    {{
        "drugName": "시타글립틴",
        "explain": "이민호님, 시타글립틴은 혈당을 조절하는 DPP-4 억제제입니다. 시타글립틴은 인크레틴 호르몬의 분해를 억제하여 인슐린 분비를 촉진하고, 글루카곤 분비를 억제하여 혈당 수치를 낮춥니다. 특히, 시타글립틴은 신장 기능이 저하된 환자에게도 용량 조절을 통해 안전하게 사용할 수 있습니다. 이 약물은 신장에서 주로 배설되기 때문에 신장질환 병력이 있는 환자에게는 적절한 용량 조절이 필요하며, 신장 기능 상태에 따라 복용량을 조절해야 합니다. 시타글립틴은 또한 혈당 변동을 최소화하여 당뇨병 관리에 큰 도움을 줍니다.",
        "effects": [
            {{"혈당 조절": "시타글립틴은 인크레틴 호르몬의 분해를 억제하여 인슐린 분비를 촉진하고, 글루카곤 분비를 억제함으로써 혈당을 조절합니다. 이는 특히 식후 혈당 상승을 효과적으로 억제하여 당뇨 관리에 도움을 줍니다."}}
        ]
    }}
  ],
  "reaction2": [
    {{
        "label": "저혈당",
        "symptom": [
            {{"현기증": "혈당이 너무 낮아질 경우 현기증을 느낄 수 있습니다."}},
            {{"피로": "저혈당으로 인해 피로감을 느낄 수 있습니다."}},
            {{"저혈당": "저혈당 증상이 나타날 수 있습니다. 이는 의식 저하, 발한, 떨림 등의 증상으로 나타날 수 있습니다."}}
        ],
        "explain": "시타글립틴은 인크레틴 호르몬의 분해를 억제하여 인슐린 분비를 촉진하고, 글루카곤 분비를 억제함으로써 혈당을 조절합니다. 그러나 이 약물은 드물게 저혈당을 유발할 수 있습니다. 저혈당이 발생하면 현기증, 피로, 떨림, 발한 등의 증상이 나타날 수 있으며, 심한 경우 의식 저하로 이어질 수 있습니다. 이민호님은 고혈압과 신장질환 병력이 있으므로 저혈당 증상이 나타날 경우 즉시 의료진과 상의해야 합니다. 저혈당 예방을 위해서는 정기적으로 혈당 수치를 모니터링하고, 약물 복용 시 음식을 함께 섭취하는 것이 좋습니다.",
        "goodHabit": [
            {{"증상 모니터링": "저혈당 증상이 나타나면 즉시 의료진과 상담하세요."}},
            {{"정기적 혈당 체크": "정기적으로 혈당을 체크하여 저혈당을 예방하세요. 저혈당 예방을 위해서는 규칙적인 식사와 약물 복용 시간을 지키는 것이 중요합니다."}}
        ]
    }}
  ],
  "totalResult": [
    {{
      "explain": "이민호님, 트라마돌과 시타글립틴을 복용할 때 신장 건강과 혈당 상태를 주의 깊게 모니터링하는 것이 중요합니다. 트라마돌은 통증 완화에 효과적이지만 중추신경계에 작용하여 졸음과 피로를 유발할 수 있으며, 시타글립틴은 인크레틴 호르몬의 분해를 억제하여 혈당을 조절하지만 저혈당을 일으킬 수 있습니다. 두 약물 모두 이민호님의 건강 상태를 고려하여 용량 조절과 모니터링이 필요합니다."
    }},
    {{
      "advice": [
        {{"건강 모니터링": "정기적으로 건강 상태를 체크하고, 이상이 있으면 즉시 의사와 상담하세요. 정기적인 혈액 검사와 신장 기능 검사가 필요합니다."}},
        {{"의사와 상담": "부작용이 발생하면 즉시 의사와 상담하세요. 의사가 필요한 조치를 제시해 줄 것입니다."}},
        {{"정기적인 검진": "정기적으로 신장 기능 검사를 받고, 혈당 수치를 체크하여 건강 상태를 관리하세요. 신장 기능 저하와 혈당 변동을 조기에 발견하고 대응하는 것이 중요합니다."}}
      ]
    }}
  ]
}}

    """

    prompt = f"""
    너는 의사를 도와주는 어시스턴트야. 입력받은 환자 정보와 시나리오, 설명유형을 기반으로 환자에게 맞춤형으로 적절한 설명을 제공해줘. 입출력 예시를 참고해서 json 형식으로 출력을 생성해줘.
    painLevel은 1부터 10까지의 정수로 표현되며, 1이 가장 낮은 수준의 통증을 의미하고 10이 가장 높은 수준의 통증을 의미해. 그리고 시나리오는 환자의 증상에 대한 처방 약물과 그 부작용, 그리고 그 부작용으로 야기될 수 있는 처방 연쇄와 그 부작용을 순서대로 포함하고 있기 때문에 이를 참고해서 설명을 생성해줘.
    예시 1 입력:

    환자 정보:
    이름: {patient_info.name}
    생년월일: {patient_info.birthdate}
    성별: {patient_info.sex}
    체중: {patient_info.weight} kg
    키: {patient_info.height} cm
    혈압: {patient_info.bloodPressure}
    과거 병력: {patient_info.pastDiseases}
    현재 복용 약물: {patient_info.currentMedications}
    알레르기: {patient_info.allergies}
    가족력: {patient_info.familyHistory}
    증상: {patient_info.symptoms}
    발병 시기: {patient_info.onset}
    통증 수준: {patient_info.painLevel}

    시나리오:
    {scenario_text}

    예시 1:
    {example}

    설명 유형: {explain_prompts[patient_info.explainType]}

    형식은 반드시 다음과 같은 형태의 json이어야 해:
    {{
      "prescription1": [
        {{
            "drugName": "...",
            "explain": "...",
            "effects": [
                {{"...": "..."}},
                {{"...": "..."}}
            ]
        }}
      ],
      "reaction1": [
        {{
            "label": "...",
            "symptom": [
                {{"...": "..."}},
                {{"...": "..."}}
            ],
            "explain": "...",
            "goodHabit": [
                {{"...": "..."}}
            ],
            "badHabit": [
                {{"...": "..."}},
                {{"...": "..."}}
            ]
        }}
      ],
      "prescription2": [
        {{
            "drugName": "...",
            "explain": "...",
            "effects": [
                {{"...": "..."}},
                {{"...": "..."}}
            ]
        }}
      ],
      "reaction2": [
        {{
            "label": "...",
            "symptom": [
                {{"...": "..."}},
                {{"...": "..."}}
            ],
            "explain": "...",
            "goodHabit": [
                {{"...": "..."}}
            ],
            "badHabit": [
                {{"...": "..."}},
                {{"...": "..."}}
            ]
        }}
      ],
      "totalResult": [
        {{"explain": "..."}},
        {{
          "advice": [
            {{"...": "..."}},
            {{"...": "..."}},
            {{"...": "..."}}
          ]
        }}
      ]
    }}

    예시 1, 2, 3을 참조하여 위의 입력 데이터를 기반으로, 위 출력 형식과 유사한 형식의 출력 데이터를 생성해 주세요.
    """

    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:wellcomlab:yagseonjido:9ogv0TRQ",
        messages=[
            {"role": "system", "content": "당신은 의사를 도와주는 어시스턴트입니다. 모든 응답은 JSON 형식으로 제공해야 합니다."},
            {"role": "user", "content": prompt}
        ],
    )

    return {response.choices[0].message.content}

@app.get("/health")
async def health():
    return {"status": "ok"}