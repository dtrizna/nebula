from nltk.tokenize import word_tokenize

# Define the input string
input_string = "The quick brown fox jumps over the lazy dog"

# Tokenize the input string
tokens = word_tokenize(input_string)

# Define the mask token
mask_token = "<mask>"

# Mask the first and last tokens
x = [mask_token if i in [0, len(tokens) - 1] else token for i, token in enumerate(tokens)]

# Create the target sequence by replacing the masked tokens with the original tokens
y = [token if i not in [0, len(tokens) - 1] else mask_token for i, token in enumerate(tokens)]

# Print the input and target sequences
print(f"X: {x}")
print(f"Y: {y}")