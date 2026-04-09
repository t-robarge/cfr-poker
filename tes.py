class Person:
    def __init__(self, name, role, classes):
        self.name = name
        self.role = role
        self.classes = classes

    def to_string(self):
        return self.__str__()
    

p = Person(name="Alice", role="student", classes=["math", "science"])
print(p.to_string())
strings = ["Hello", "World", "This", "Is", "A", "Test"]
s = List[strings]

def func() -> str:
    return "This is a function that returns a string."