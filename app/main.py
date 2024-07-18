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
    너는 의사를 도와주는 어시스턴트야. 입력받은 환자 정보를 바탕으로 적절한 처방약, 부작용, 그리고 그 부작용이 초래할 수 있는 연쇄 처방들을 노드와 엣지 형태로 제공하고 총 정리까지 생성해줘.
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

    예시1을 참조하여 위의 입력 데이터를 기반으로, 위 출력 형식과 유사한 형식의 출력 데이터를 생성해 주세요.
    """
    response = client.chat.completions.create(
        model="ftjob-BGK5FrqtvrsxzQXwCeMOqD3R",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
    )

    return {response.choices[0].message.content}

@app.post("/simulation/patient/")
async def simulate_patient(patient_info: SimulateResult):
    explain_prompts = {
        1: "환자들이 이해할 수 있도록 쉽고 친절하게 설명해주세요.",
        2: "간단하고 핵심적인 정보만 제공해 주세요.",
        3: "전문가 수준의 자세하고 깊이있는 정보를 제공해 주세요."
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
        {{"drugName": "타이레놀"}},
        {{
          "explain": "타이레놀은 두통과 발열을 완화하는 데 도움을 주는 약입니다. 타이레놀은 아세트아미노펜 성분을 포함하고 있으며, 이 성분은 통증을 줄이고 열을 내리는 데 탁월합니다. 현재 두통과 복통으로 고생하고 계시기 때문에 타이레놀이 증상을 완화하는 데 효과적일 것입니다. 하지만 매일 100mg의 아스피린을 복용 중이시므로 타이레놀을 과다 복용하지 않도록 주의해주세요."
        }},
        {{
          "effects": [
            {{"통증 완화": "타이레놀은 아픈 느낌을 줄여줍니다. 특히 두통과 같은 일상적인 통증을 완화하는 데 효과적입니다."}},
            {{"해열 작용": "타이레놀은 열을 낮추어 체온을 정상으로 유지합니다. 발열로 인한 불편함을 줄여줍니다."}}
          ]
        }}
      ],
      "reaction1": [
        {{"label": "간 기능 저하"}},
        {{
          "symptom": [
            {{"황달": "피부나 눈이 노래질 수 있습니다. 이는 간 기능이 저하되어 빌리루빈이 축적되기 때문입니다."}},
            {{"어두운 소변": "소변 색이 어두워질 수 있습니다. 이는 간 기능이 저하되어 빌리루빈이 소변으로 배출되기 때문입니다."}}
          ]
        }},
        {{
          "explain": "홍길동님, 타이레놀을 많이 복용하면 간이 손상될 수 있어요. 과거에 고혈압과 당뇨를 앓으셨기 때문에 간 건강에 특히 주의가 필요합니다. 간 기능이 저하되면 황달이나 어두운 소변과 같은 증상이 나타날 수 있습니다."
        }},
        {{
          "goodHabit": [
            {{"복용약 준수": "의사나 약사가 정해준 양을 꼭 지켜서 복용해야 해요. 복용량을 준수하면 부작용을 줄일 수 있습니다."}}
          ]
        }},
        {{
          "badHabit": [
            {{"과다 복용 금지": "정해진 양보다 많이 복용하지 않도록 주의하세요. 과다 복용은 간 손상의 위험을 높입니다."}},
            {{"알코올 섭취 자제": "약을 복용하는 동안에는 술을 마시지 마세요. 알코올은 간에 추가적인 부담을 줄 수 있습니다."}}
          ]
        }}
      ],
      "prescription2": [
        {{"drugName": "실리마린"}},
        {{
          "explain": "실리마린은 간을 보호하고 건강하게 유지하는 데 도움을 주는 약입니다. 실리마린은 밀크씨슬에서 추출한 성분으로, 간 세포를 보호하고 재생을 촉진합니다. 과거에 고혈압과 당뇨를 앓으셨기 때문에 실리마린이 간 건강을 유지하는 데 도움이 될 것입니다."
        }},
        {{
          "effects": [
            {{"간 보호": "실리마린은 간 세포의 재생을 돕고, 간 손상을 예방합니다. 특히, 간이 손상된 경우에도 간 기능을 회복시키는 데 도움을 줍니다."}},
            {{"항산화 작용": "실리마린은 항산화 작용을 통해 세포를 손상으로부터 보호합니다. 자유 라디칼을 제거하여 세포 손상을 방지합니다."}}
          ]
        }}
      ],
      "reaction2": [
        {{"label": "위장관 문제"}},
        {{
          "symptom": [
            {{"복통": "실리마린을 복용하면 위장관에 자극을 주어 배가 아플 수 있습니다."}},
            {{"속쓰림": "실리마린이 위산 분비를 자극하여 속이 쓰릴 수 있습니다."}}
          ]
        }},
        {{
          "explain": "홍길동님, 실리마린을 복용하면 배가 아프거나 속이 쓰릴 수 있어요. 이는 약물이 위장 점막을 자극하기 때문입니다. 음식을 먹고 약을 복용하면 이런 증상을 줄일 수 있습니다."
        }},
        {{
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
        {{"drugName": "..."}},
        {{"explain": "..."}},
        {{
          "effects": [
            {{"...": "..."}},
            {{"...": "..."}}
          ]
        }}
      ],
      "reaction1": [
        {{"label": "..."}},
        {{
          "symptom": [
            {{"...": "..."}},
            {{"...": "..."}}
          ]
        }},
        {{"explain": "..."}},
        {{
          "goodHabit": [
            {{"...": "..."}}
          ]
        }},
        {{
          "badHabit": [
            {{"...": "..."}},
            {{"...": "..."}}
          ]
        }}
      ],
      "prescription2": [
        {{"drugName": "..."}},
        {{"explain": "..."}},
        {{
          "effects": [
            {{"...": "..."}},
            {{"...": "..."}}
          ]
        }}
      ],
      "reaction2": [
        {{"label": "..."}},
        {{
          "symptom": [
            {{"...": "..."}},
            {{"...": "..."}}
          ]
        }},
        {{"explain": "..."}},
        {{
          "goodHabit": [
            {{"...": "..."}}
          ]
        }},
        {{
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

    예시1을 참조하여 위의 입력 데이터를 기반으로, 위 출력 형식과 유사한 형식의 출력 데이터를 생성해 주세요.
    """

    response = client.chat.completions.create(
        model="ftjob-BGK5FrqtvrsxzQXwCeMOqD3R",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
    )

    return {response.choices[0].message.content}

@app.get("/health")
async def health():
    return {"status": "ok"}