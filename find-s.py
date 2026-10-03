# FIND-S Algorithm
# Name: S.ALHUMAITHE
# Reg No: 192524292

# Training data
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Initialize the most specific hypothesis
hypothesis = ['Ø', 'Ø', 'Ø', 'Ø', 'Ø', 'Ø']

print("Training Examples:")
for row in data:
    print(row)

# FIND-S algorithm
for example in data:
    attributes = example[:-1]
    target = example[-1]

    # Consider only positive examples
    if target == 'Yes':
        for i in range(len(hypothesis)):
            if hypothesis[i] == 'Ø':
                hypothesis[i] = attributes[i]
            elif hypothesis[i] != attributes[i]:
                hypothesis[i] = '?'

        print("\nAfter positive example:", attributes)
        print("Hypothesis:", hypothesis)

print("\nFinal Most Specific Hypothesis:")
print(hypothesis)