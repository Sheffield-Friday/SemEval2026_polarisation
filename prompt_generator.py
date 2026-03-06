from jinja2.exceptions import TemplateError
from transformers import AutoTokenizer
from typing import Optional


class PromptGenerator:

    @staticmethod
    def single_label_classes_bullet(
        classes: list, intro_string: str = "Classes:", bullet: str = "*"
    ):
        """Create bullet list for single-label list of classes without descriptions.

        Args:
            classes (List[str]): list of class names.
            intro_string (str): string used to introduce the list;
                the default is 'Classes:'. This will always be proceeded
                by the newline character.
            bullet (str): string to be used as the bullet point;
            default is '*'.

        Returns:
            str: the prompt text for the class list.
        """
        prompt_text = intro_string + "\n"
        for clss in classes:
            prompt_text += f"{bullet} {clss}\n"
        return prompt_text

    @staticmethod
    def single_label_classes_list(
        classes: list, intro_string: str = "Classes:", separator: str = ", "
    ):
        """Create a simple list of single-label classes without description.

        Args:
            classes (List[str]): list of class names.
            intro_string (str): string used to introduce the list;
                the default is 'Classes:'. This will always be proceeded
                by a space character.
            separator (str): string used to separate two classes.
        """
        prompt_text = intro_string + " "
        for i, clss in enumerate(classes):
            prompt_text += clss
            if i < (len(classes) - 1):
                prompt_text += separator
        prompt_text += "."

        return prompt_text

    @staticmethod
    def build_system_prompt(prompt_template: str, class_text: str = ""):
        return prompt_template.format(CLASSES=class_text)

    @staticmethod
    def build_prompt(prompt_template: str, input_text: str, class_text: str = ""):
        return prompt_template.format(INPUT_TEXT=input_text, CLASSES=class_text)

    @staticmethod
    def create_full_prompt(
        tokenizer: AutoTokenizer, user_prompt: str, system_prompt: Optional[str] = None
    ):
        if not system_prompt:
            return [{"role": "user", "content": user_prompt}]

        system_message = {"role": "system", "content": system_prompt}
        try:
            tokenizer.apply_chat_template([system_message])
            user_message = {"role": "user", "content": user_prompt}
            messages = [system_message, user_message]
        except (TemplateError, ValueError):
            combined_message = f"**System instruction start**\n{system_prompt}\n**End of system instruction**\n\n{user_prompt}"
            messages = [{"role": "user", "content": combined_message}]

        return messages

        # return tokenizer.apply_chat_template(
            # messages, tokenize=False, add_generation_prompt=True
        # )

