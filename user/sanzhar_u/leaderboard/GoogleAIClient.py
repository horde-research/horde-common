from typing import List, Dict, Any, Optional

import httpx

class GoogleAIClient():
    def __init__(self,
                 api_key:str,
                 proxy: Optional[str] = None,
                 timeout: int = 60):
        """
        Args:
            api_key: API ключ для доступа к внешнему API.
            proxy: Прокси для доступа к внешнему источнику.
            timeout: Таймаут в секундах
        """

        self.api_key = api_key
        self.client = self.setup_client(proxy=proxy, timeout=timeout)
    
    def setup_client(self, proxy: Optional[str], timeout: int) -> None:
        """
        Настройка синхронного HTTP клиента для Google AI.

        Args:
            proxy: Прокси для доступа к внешнему источнику.
            timeout: Таймаут в секундах
        """
        proxies = {'http://': proxy, 'https://': proxy} if proxy else None
        timeout = httpx.Timeout(timeout, connect=timeout)
        client = httpx.Client(timeout=timeout, verify=False)

        return client

    def _perform_request(self, 
                        url: str, 
                        headers: Optional[dict] = None, 
                        json: Optional[dict] = None, 
                        params: Optional[dict] = None) -> httpx.Response:
        """
        Выполнение HTTP запроса с использованием настроенного клиента.

        Args:
            url:
                Ссылка API
            headers:
                Заголовки http запроса
            json:
                Данные для POST запроса
            params:
             Другие параметры http запроса
            
        Returns:
            Ответ от API

        Raises:
            Если возникла ошибка при отправке/обработке запроса
        """

        try:
            response = self.client.request(method="POST", 
                                            url=url, 
                                            headers=headers, 
                                            json=json, 
                                            params=params)
            
            response.raise_for_status()
            return response
        except httpx.HTTPError as e:
            raise RuntimeError(f"HTTP error: {str(e)}")

    def build_headers(self) -> Dict[str, str]:
        """ 
        Строит headers для реквестов.

        Returns:
            headers для реквестов
        """
        return {"Content-Type": "application/json"}

    def build_text_content(self, prompt: str, system_message: Optional[str]) -> str:
        """
        Создает текстовый контент, включая системное сообщение и запрос.

        Args:
            prompt: Текст запроса.
            system_message: Опциональное системное сообщение.

        Returns:
            Сформированный текстовый контент.
        """

        text_content = f"Системные настройки: {system_message}\n" if system_message else ""
        text_content += f"Текст: {prompt}"
        return text_content


    def build_payload(self, 
                        text_content: str, 
                        chat_history: Optional[List[Dict[str, Any]]], 
                        max_tokens: int, 
                        temperature: float) -> Dict[str, Any]:
        """
        Строит пейлоуд для запросов.

        Args:
            text_content:
                Передаваемый текст
            chat_history:
                История чатов
            max_tokens:
                Параметр модели макс кол-во токенов ответа
            temperature:
                Параметр модели температура
        
        Returns:
            Сформированный пейлоуд для запросов
        """
        payload = {
            "contents": [
                {
                    "parts": [{"text": text_content}],
                    "role": 'user',
                },
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
            ],
        }

        if chat_history is not None:
            for entry in chat_history:
                transformed_entry = {
                    'parts': {'text': entry['text']},
                    'role': entry['role']
                }

                payload['contents'].insert(-1, transformed_entry)

        return payload

    def build_params(self) -> Dict[str, str]:
        """ 
        Оборачивает апи ключ в словарь.

        Returns:
            Апи ключ в словаре
        """
        return {'key': self.api_key}

    def build_url(self, model: str) -> str:
        """
        Вставляет модель в юрл для запросов.
        
        Args:
            model:
                Наименование модели
        
        Returns:
            юрл для запросов
        """
        return f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"

    def process_response(self, response: Any) -> Dict[str, Any]:
        """
        Парсит респонс от ЛЛМ.
        
        Args:
            response:
                Результат запроса

        Returns:
            Распарсенный ответ от ЛЛМ
        """
        json_response = response.json()
        text = json_response["candidates"][0]['content']['parts'][0]['text']
        return {'text': text, 'metadata': json_response}

    def completions_request(self,
                            prompt: str,
                            chat_history: List[dict] = None,
                            system_message: str = None,
                            temperature: float = 0.2,
                            max_tokens: int = 2048,
                            model: str = 'gemini-1.5-flash') -> Dict[str, Any]:
        """
        Отправляет запрос на генерацию текста на основе предоставленного запроса.

        Args:
            prompt: Входной текст.
            chat_history: Необязательная история чата.
            system_message: Необязательное системное сообщение, описывающее задачу.
            temperature: Параметр температуры модели.
            max_tokens: Максимальное количество токенов для ответа.
            model: Версия модели.

        Returns:
            Словарь с сгенерированным текстом и полной метаинформацией о ответе.
        """

        headers = self.build_headers()
        text_content = self.build_text_content(prompt, system_message)
        payload = self.build_payload(text_content, chat_history, max_tokens, temperature)
        params = self.build_params()
        url = self.build_url(model)

        response = self._perform_request(url, headers=headers, json=payload, params=params)
        return self.process_response(response)



    def get_embedding(self, text: str, model: str = 'embedding-001') -> List[float]:
        """
        Получение векторного представления текста с помощью модели векторизации.

        Args:
            text:
                Передаваемый текст
            model:
                Версия модели

        Returns:
            Список с векторами текста
        """

        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "content": {                                       
                "parts": [{"text": text}],
                "role": "user"
            }
        }

        params = {
            'key': self.api_key
        }

        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:embedContent"

        response = self._perform_request(url, headers=headers, json=payload, params=params)
        
        embedding = response.json()['embedding']['values']
        if not embedding:
            raise KeyError("Response json doesn't have keys 'embedding:values'")

        return embedding
