# import dane
# import dane as dn
from dane import nrfilii as nf,book as bk
from moje_funkcje.collfunction import czytaj_slownik,czytaj_liste

#CTRL+D --> powielenie linii/bloku tekstu
#CTRL + / --> komentowanie/odkomwntowanie bloku tekstu
print("________________ bezpośrenio z dane ________________")
print(nf)
print(bk)

print("________________ użycie funkcji ________________")
czytaj_liste(nf)
czytaj_slownik(bk)
