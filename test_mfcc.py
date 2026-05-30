from audio.feature_extractor import extract_mfcc

mfcc = extract_mfcc("audio.wav")

print("Shape:", mfcc.shape)
print(mfcc)