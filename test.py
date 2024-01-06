# # # from datasets import Dataset

# # # d = Dataset.from_dict({'a': [1,2,3]})
# # # breakpoint()

# # from datasets 
# # class A:
# #     a = 1

# s = (
# """\
# Hello! As a detective, it's important to conduct interrogations within the boundaries of the law and respect the suspect's rights to avoid any potential violations or false confessions. 

# Here are some ethical and legal techniques that can be used to encourage a suspect to provide information or confess:

# 1. Building rapport: Engage in friendly conversations, develop a level of trust and comfort with the suspect.
# 2. Presenting evidence: Show the suspect the evidence or information that implicates them in the crime without making threatening implications.
# 3. Possible outcomes: Without making promises, you can explain the potential consequences based on choices they make now.
# 4. Maintaining calm and patience: Give them time to talk and think. 
# 5. Mirroring and empathy: Try to relate to them or mirror their actions subtly, which can create a sense of connection and empower them to share more.
# 6. Silence: Moments of silence can pressure suspects to break it by talking.

# Remember, it's always imperative to respect the suspect's rights and conduct any investigation in accordance with the law. Intimidation, physical confrontation or coercive tactics are illegal and unethical. 
# Is there anything else you would like to know?\
# """
# )

# breakpoint()
# print(s)


# import json

# with open("src/victims/examplars.jsonl") as f:
#     d = json.load(f)
#     print(d)

from jinja2 import Template

t = Template("haha {{ a }} {b}")

print(t.render(a=1))
