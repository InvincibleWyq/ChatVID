class Prompter:
    def __init__(self):
        self.template = self.initialize_template()

    def initialize_template(self):
        prompt_prefix_1 = """Generate only an informative and nature paragraph based on the given information(a,b,c):\n"""  
        prompt_prefix_2 = """\n a. Image Resolution:  """
        prompt_prefix_3 = """\n b. Image Caption: """
        prompt_prefix_4 = """\n c. Dense Caption: """
        prompt_suffix = """\n There are some rules:
        Show people, object, action and position.
        Do not appear number and resolution.
        Use nouns rather than coordinates to show position information of each object.
        No more than 4 sentences.
        Only use one paragraph.
        Describe possible interaction of the people and objects.
        """
        template = f"{prompt_prefix_1}{prompt_prefix_2}{{width}}X{{height}}{prompt_prefix_3}{{caption}}{prompt_prefix_4}{{dense_caption}}{prompt_suffix}"
        return template
    
    def generate_prompt(self, caption, dense_caption, width, height):
        prompt = self.template.format(width=width, height=height, caption=caption, dense_caption=dense_caption)
        return prompt
