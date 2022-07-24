""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict

_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
# symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet

vowels = list(range(0X0905, 0X0915)) + [0X0960, 0X0961]
consonants = list(range(0X0915, 0X0929)) + list(range(0X092A, 0X0931)) + [0X0932, 0X0933] + list(range(0X0935, 0X093A))
punct = list(range(0X0901, 0X0904)) + list(range(0X093C, 0X094E)) + [0X0964, 0X0965]
numbers = [0X0950, 0X0AF1] + list(range(0X0966, 0X0971))
pronounced = list(range(0X0958, 0X0960))
extra_vowels = [0X0962]
charecters = vowels + consonants + punct + numbers + pronounced + extra_vowels
_letters = [chr(ch) for ch in charecters]
marathi_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters)
