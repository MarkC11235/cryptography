text = """
The first known recorded explanation of frequency analysis (indeed, of any kind of cryptanalysis) was given in the 9th century by Al-Kindi, an Arab polymath, in A Manuscript on Deciphering Cryptographic Messages.[3] It has been suggested that a close textual study of the Qur'an first brought to light that Arabic has a characteristic letter frequency.[4] Its use spread, and similar systems were widely used in European states by the time of the Renaissance. By 1474, Cicco Simonetta had written a manual on deciphering encryptions of Latin and Italian text.[5]

Several schemes were invented by cryptographers to defeat this weakness in simple substitution encryptions. These included:

Homophonic substitution: Use of homophones — several alternatives to the most common letters in otherwise monoalphabetic substitution ciphers. For example, for English, both X and Y ciphertext might mean plaintext E.
Polyalphabetic substitution, that is, the use of several alphabets — chosen in assorted, more or less devious, ways (Leone Alberti seems to have been the first to propose this); and
Polygraphic substitution, schemes where pairs or triplets of plaintext letters are treated as units for substitution, rather than single letters, for example, the Playfair cipher invented by Charles Wheatstone in the mid-19th century.
A disadvantage of all these attempts to defeat frequency counting attacks is that it increases complication of both enciphering and deciphering, leading to mistakes. Famously, a British Foreign Secretary is said to have rejected the Playfair cipher because, even if school boys could cope successfully as Wheatstone and Playfair had shown, "our attachés could never learn it!".

The rotor machines of the first half of the 20th century (for example, the Enigma machine) were essentially immune to straightforward frequency analysis. However, other kinds of analysis ("attacks") successfully decoded messages from some of those machines.[6]

Frequency analysis requires only a basic understanding of the statistics of the plaintext language and some problem-solving skills, and, if performed by hand, tolerance for extensive letter bookkeeping. During World War II, both the British and the Americans recruited codebreakers by placing crossword puzzles in major newspapers and running contests for who could solve them the fastest. Several of the ciphers used by the Axis powers were breakable using frequency analysis, for example, some of the consular ciphers used by the Japanese. Mechanical methods of letter counting and statistical analysis (generally IBM card type machinery) were first used in World War II, possibly by the US Army's SIS. Today, the work of letter counting and analysis is done by computer software, which can carry out such analysis in seconds. With modern computing power, classical ciphers are unlikely to provide any real protection for confidential data.
"""

filtered_text = ""
for char in text:
    if 'a' <= char.lower() <= 'z':
        filtered_text += char.lower()

filtered_text_length = len(filtered_text)

letter_counts = {}
for char in filtered_text:
    if char in letter_counts:
        letter_counts[char] += 1
    else:
        letter_counts[char] = 1

frequencies = {}
for letter in "abcdefghijklmnopqrstuvwxyz":
    count = letter_counts.get(letter, 0)
    frequencies[letter] = (count / filtered_text_length) * 100 if filtered_text_length > 0 else 0

print("Total characters:", filtered_text_length)
for letter in sorted(frequencies):
    print(letter + ":", "%.2f" % frequencies[letter] + "%", f"({letter_counts.get(letter, 0)} occurrences)")