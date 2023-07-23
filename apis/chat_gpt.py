import os
import openai


def completion():
    openai.organization = "org-xmxpTocKt0WeQK9Gec2UXfgS"
    openai.api_key = "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"#os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Translate the following into spanish and use html encoding for special characters: With more choices, more flexibility and more protection than "
               "ever, there’s no better time to explore all that the new Essence Healthcare Medicare Advantage plans "
               "have to offer.",
        max_tokens=1000,
        temperature=0
    )
    print(response)


completion()
