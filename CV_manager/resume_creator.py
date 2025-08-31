from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

model_id = "qwen3:4b-instruct-2507-q8_0"

class ResumeCreator(BaseModel):
    name: str
    position: str
    email: str
    phone: str
    address: str
    text: str

    def create_resume(self):
        """
        Creates a formatted resume based on the provided personal details and resume text.

        This method uses a defined chat prompt template and a machine learning model to generate
        a resume for the user. The resume creation process involves formatting the user's details into
        a structured prompt and passing it to the model.

        :param self: An instance of the class containing the necessary user information.
        :return: Generated resume text as a string.
        :rtype: str
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that creates a resume based on the provided text."),
            ("user", "Create a resume for {name} as a {position} with the following information:\n"
                    "Email: {email}\n"
                    "Phone: {phone}\n"
                    "Address: {address}\n"
                    "Text: {text}"),
            ("assistant", "Here is your resume:\n")
        ])
        model = ChatOllama(model=model_id, temperature=0.7)
        text_prompt = prompt.format(name=self.name,
                                    position=self.position,
                                    email=self.email,
                                    phone=self.phone,
                                    address=self.address,
                                    text=self.text)
        response = model.invoke(text_prompt)
        return response.content

    def __str__(self):
        return f"Resume for {self.name} at {self.position}"

    def __repr__(self):
        return (f"ResumeCreator(name={self.name}, position={self.position}, email={self.email}, "
                f"phone={self.phone}, address={self.address}, text={self.text})")

