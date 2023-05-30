completo = []
for l in range(90):
    for c in range(90):
        completo.append(1)
print(f"Completo: {len(completo)}")

metade = []
for l in range(1,90):
    for c in range(l):
        metade.append(1)
print(f"Metade: {len(metade)}")
