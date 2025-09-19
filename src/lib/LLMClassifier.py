import requests
import json
import time
import numpy as np
from typing import List, Dict, Optional
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

class LLMClassifier:
    """
    Generic multi-label classifier using different LLM providers
    """
    
    def __init__(self, 
                 provider: str = "openai", 
                 model_name: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize the LLM classifier
        
        Args:
            provider: "openai", "ollama", "huggingface", "vertex_ai"
            model_name: Name of the model to use
            api_key: API key (if needed)
            base_url: Base URL for custom APIs
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        
        # Initialize the client according to the provider
        if provider == "openai":
            from openai import OpenAI
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                self.client = None  # Will be initialized later when api_key is provided
        elif provider == "ollama":
            self.ollama_url = base_url or "http://localhost:11434"
        
        self.prompt_base = """You are an AI Assistant specialized in multi-label classification for StackOverflow questions.

TASK: Analyze questions and predict exactly 5 tags for each one.

STRICT RULES:
1. Respond ONLY in valid JSON format
2. Do not add any text before or after the JSON
3. Do not provide explanations or advice

REQUIRED OUTPUT FORMAT:
```json
{
  "predictions": [
    {
      "Id": [QUESTION_ID],
      "TagsList": ["tag1", "tag2", "tag3", "tag4", "tag5"]
    }
  ]
}
```

IMPORTANT: Respond only with the JSON, nothing else.
"""
        
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Generic call to LLM according to provider"""
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    return self._call_openai(prompt)
                elif self.provider == "ollama":
                    return self._call_ollama(prompt)
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
                    
            except Exception as e:
                print(f"Error {self.provider} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
        print("Failed after all retries")
        return ""

    def _call_openai(self, prompt: Dict[str, str]) -> str:
        """Call to OpenAI API"""
        if self.client is None:
            if self.api_key:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            else:
                raise ValueError("OpenAI API key not provided. Please set api_key when initializing or set OPENAI_API_KEY environment variable.")
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt.get("user_prompt")},
                {"role": "system", "content": prompt.get("system_prompt")}
            ],
            temperature=0.1,
            max_tokens=500,
            top_p=0.5
        )
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""

    def _call_ollama(self, prompt: Dict[str, str]) -> str:
        format = {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Id": {"type": "integer"},
                            "TagsList": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["Id", "TagsList"]
                    }
                }
            },
            "required": ["predictions"]
        }

        """Call to Ollama local"""
        payload = {
            "model": self.model_name,
            "prompt": prompt.get("user_prompt"),
            "system": prompt.get("system_prompt"),
            "stream": False,
            "options": {
                "temperature": 0.1,
                "seed": 42,
                "top_k": 10,
                "top_p": 0.5
            }
        }

        if(self.model_name == "mistral:latest"):
            payload["format"] = format
        
        response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '').strip()

    def _create_structured_prompt(
        self,
        questions: pd.DataFrame,
        prompt_type: str = "zero_shot",
        examples: pd.DataFrame = None,
        available_tags: List[str] = None
    ) -> Dict[str, str]:
        """Create a structured prompt with forced JSON format"""
        prompt = {
            "system_prompt": self.prompt_base,
            "user_prompt": ""
        }
        # if "zero_shot" system prompt is already initialized        
        if prompt_type == "few_shot" and examples is not None:

            prompt["system_prompt"] += "EXAMPLES OF QUESTIONS AND THEIR TAGS:\n\n"
            
            for i, example in examples.iterrows():
                prompt["system_prompt"] += f"Id:{example.Id}\nTitle:{example.Title}\nBody:{example.Body}\n"
                prompt["system_prompt"] += f"TagsList:{example.TagsList}\n\n"

        elif prompt_type == "tags_list" and available_tags is not None:
            prompt["system_prompt"] += f"YOU MUST CHOOSE 5 TAGS FROM THE FOLLOWING LIST:\n{', '.join(available_tags)}\n\n"

        # Add questions to the user prompt
        prompt["user_prompt"] = f"QUESTION TO ANALYZE AND PREDICT TAGS:\n\n"
        for _, q in questions.iterrows():
            prompt["user_prompt"] += f"Id:{q.Id}\nTitle:{q.Title}\nBody:{q.Body}\n\n"
        
        return prompt

    def predict(
        self,
        questions: pd.DataFrame,
        prompt_type: str = "zero_shot",
        examples: pd.DataFrame = None,
        available_tags: List[str] = None
    ) -> str:
        """Predict tags for a list of questions"""
        prompt = self._create_structured_prompt(questions, prompt_type, examples, available_tags)
        response = self._call_llm(prompt)
        return response

    def _parse_json_response(self, response: str) -> pd.DataFrame:
        """Parse the JSON response"""
        try:
            # Clean the response by removing markdown tags
            cleaned_response = response.strip()
            
            # Remove ```json at the beginning and ``` at the end
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
                
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
                
            cleaned_response = cleaned_response.strip()
            
            # Parse the JSON
            data = json.loads(cleaned_response)

            # Extract predictions
            predictions = data.get('predictions', [])
            
            return pd.DataFrame(predictions)
            
        except json.JSONDecodeError as e:
            print(f"Error during JSON deserialization: {response}")
            print(f"Detailed error: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error during parsing: {e}")
            return pd.DataFrame()
    
    def predict_and_transform(
        self,
        questions: pd.DataFrame,
        mlb: MultiLabelBinarizer,
        prompt_type: str = "zero_shot",
        examples: pd.DataFrame = None,
        available_tags: List[str] = None
    ) -> np.ndarray:
        """
        Predict tags and transform them into binary format compatible with MultiLabelBinarizer
        """
        # Get predictions
        response = self.predict(questions, prompt_type, examples, available_tags)
        predictions_df = self._parse_json_response(response)
        
        # Check that 'TagsList' and 'Id' columns exist in the DataFrame
        if predictions_df.empty or 'TagsList' not in predictions_df.columns or 'Id' not in predictions_df.columns:
            # Return a matrix of zeros if no predictions or missing columns
            return np.zeros((len(questions), len(mlb.classes_)))
        
        # Filter predicted tags to keep only those in mlb.classes_
        filtered_tags = []
        for tags_list in predictions_df['TagsList']:
            if isinstance(tags_list, list):
                filtered_tags.append([tag for tag in tags_list if tag in mlb.classes_])
            else:
                filtered_tags.append([])
        
        # Transform with MultiLabelBinarizer
        y_pred_binary = mlb.transform(filtered_tags)
        
        return y_pred_binary