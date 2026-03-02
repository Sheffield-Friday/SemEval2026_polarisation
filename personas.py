"""
The categories in the polarisation types are:

Political/ideological polarization
Racial or ethnic polarization
Religious polarization
Gender polarization / sexual orientation polarization
Other

From this we derive 5 characteristics to use for persona-building - political affilication, race, religion, gender and sexual orientation

To avoid creating too many possible personas, the choices don't cover all potential options for these characteristics. 
Future work could examine the effect of including more choices or adding additional characteristics.
"""

import random

attributes = {
    "politics": ["left-wing", "politically moderate", "right-wing"],
    "race": ["Black", "East Asian", "Middle Eastern", "Mixed Race", "White"],
    "religion": ["Atheist", "Christian", "Hindu", "Jewish", "Muslim"],
    "gender": ["man", "woman"],
    "orientation": ["heterosexual", "homosexual"]
}

class Person:
    def __init__(self, attrs={}):
        self.politics =     attrs['politics'] if 'politics' in attrs.keys() else           random.choice(attributes["politics"])
        self.race =         attrs['race'] if 'race' in attrs.keys() else                   random.choice(attributes["race"])
        self.religion =     attrs['religion'] if 'religion' in attrs.keys() else           random.choice(attributes["religion"])
        self.gender =       attrs['gender'] if 'gender' in attrs.keys() else               random.choice(attributes["gender"])
        self.orientation =  attrs['orientation'] if 'orientation' in attrs.keys() else     random.choice(attributes["orientation"])

        self.orientation_gender = f"{self.orientation} {self.gender}"

    def __str__(self):
        return f"{self.politics} {self.race} {self.religion} {self.orientation} {self.gender}"
    

def generate_list_of_people(num_of_people):
    """
    Generates N people
    Returns a set that tries to give even distribution over the attributes
    """

    people = []

    # set up counts data structure
    counts = {}
    for attr in attributes.keys():
        counts[attr] = dict.fromkeys(attributes[attr], 0)

    for i in range(num_of_people):

        # generate the next person
        if i < num_of_people / 2:
            # generate first half of the people randomly
            person = Person()
            
        else:
            # get the rarest attributes
            person_attrs = {}
            for attr in attributes.keys():
                    # find the choices with the lowest counts and make a random choice between them
                    min_attr_count = min(counts[attr].values())
                    choices = [x for x in counts[attr].keys() if counts[attr][x] == min_attr_count]
                    person_attrs[attr] = random.choice(choices)

                    # person_attrs[attr] = min(counts[attr], key = counts[attr].get) # <- don't use, always selects first key in a draw so will end up imbalanced over time

            # generate person with those attributes
            person = Person(attrs=person_attrs)
        
        people.append(person)

        # increment the attribute counts
        counts['politics'][person.politics] += 1
        counts['race'][person.race] += 1
        counts['religion'][person.religion] += 1
        counts['gender'][person.gender] += 1
        counts['orientation'][person.orientation] += 1
    
    return people, counts


if __name__ == "__main__":

    # example to show how the generation works
    print("Generating one person:")
    person = Person()
    print(person)

    print("Generating a set of people:")
    person_list, attr_counts = generate_list_of_people(10)
    print(person_list, attr_counts)

    for person in person_list:
        print(person)
