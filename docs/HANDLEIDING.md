# Nederlandse handleiding — Krea2 Two Character Composer

Met deze ComfyUI-node maak je één Krea 2-afbeelding met twee personages, ieder
met een eigen prompt en karakter-LoRA. Verdeel het beeld links/rechts,
boven/onder, met twee vrij te plaatsen kaders of met aangesloten maskers.

De regionale engine zit in deze uitbreiding. Je hoeft **ComfyUI-Krea2-Regional
niet apart te installeren**. Deze versie ondersteunt Krea 2, niet Z-Image Turbo.

## 1. Installeren

1. Gebruik een recente ComfyUI-versie met ondersteuning voor Krea 2. In
   CLIPLoader moet je het type `krea2` kunnen kiezen.
2. Download de installatie-ZIP van de
   [releasepagina](https://github.com/cyberdeliaAI/comfyui-krea2-two-character-composer/releases/tag/v0.4.5).
   Pak de map uit in `ComfyUI/custom_nodes`.
3. Controleer dat `__init__.py` direct in
   `custom_nodes/comfyui-krea2-two-character-composer` staat. Een tweede map met
   dezelfde naam daarbinnen is een verkeerde installatie.
4. Zet je **Krea 2-karakter-LoRA’s** in `ComfyUI/models/loras`.
5. Herstart ComfyUI, vernieuw de browser en zoek **Krea2 Two Character Composer**.

Je kunt ook vanuit `ComfyUI/custom_nodes` installeren met:

```sh
git clone https://github.com/cyberdeliaAI/comfyui-krea2-two-character-composer.git
```

Er zijn geen extra Python-pakketten nodig naast je werkende ComfyUI-installatie.
Voor negatieve prompts is de apart geïnstalleerde
[NegPiP-uitbreiding](https://github.com/cyberdeliaAI/comfyui-negpip-zimage)
versie 2.1 of nieuwer nodig. Zonder negatieve prompts is NegPiP niet nodig.

## 2. Aansluiten

Begin bij voorkeur met een Krea 2-workflow waarmee je al kunt genereren. Of
sleep de [voorbeeldworkflow](../example_workflows/two_characters.json) in ComfyUI
en kies je eigen model, tekstencoder, VAE en LoRA’s.

| Van | Naar |
|---|---|
| Krea 2-model | Composer → `model` ingang |
| CLIPLoader met type `krea2` | Composer → `clip` ingang |
| EmptySD3LatentImage | Composer → `latent` ingang |
| Composer → `model` uitgang | KSampler → `model` |
| Composer → `positive` uitgang | KSampler → `positive` |
| Composer → `negative` uitgang | KSampler → `negative` |
| Composer → `latent` uitgang | KSampler → `latent_image` |
| KSampler → `samples` | VAE Decode → `samples` |
| VAE Decode → `IMAGE` | Save Image, of Save Image With Metadata |

Gebruik de **modeluitgang van de Composer** voor de sampler: daarin zit de
regionale LoRA-aansturing. De voorbeeldworkflow begint voor Turbo met 8 stappen
en CFG 1. Voor RAW gebruik je de instellingen van je werkende RAW-workflow.
De Composer stelt de sampler zelf niet in.

De optionele uitgang `prompts` kun je aan Preview Any koppelen om de verwerkte
prompts en instellingen te bekijken. Met `region_masks` → Mask To Image →
Preview Image kun je beide maskers bekijken.

## 3. Eerste afbeelding: links 60%, rechts 40%

Vul bijvoorbeeld het volgende in. De triggerwoorden en LoRA-bestanden zijn
voorbeelden: gebruik de waarden die bij jouw LoRA’s horen.

| Instelling | Voorbeeld |
|---|---|
| Base prompt | `Two adults in a garden, daylight, illustration` |
| Character 1 trigger | Het triggerwoord van LoRA 1 |
| Character 1 prompt | `a woman with blonde hair, red coat, standing` |
| Character 1 LoRA | Jouw eerste karakter-LoRA |
| Character 1 LoRA strength | Begin met de aanbevolen sterkte van die LoRA |
| Character 2 trigger | Het triggerwoord van LoRA 2 |
| Character 2 prompt | `a man with black hair, blue jacket, sitting` |
| Character 2 LoRA | Jouw tweede karakter-LoRA |
| Direction | `Left / Right` |
| Character 1 percent | `60` |
| Separation | `Balanced` |
| LoRA edge softness | `0` |
| Base strength | `1.0` |

Karakter 1 gebruikt links ongeveer 60% van het beeld, karakter 2 rechts 40%.
De grens wordt op het 16px-raster van het model gelegd. Kies `Top / Bottom`
om karakter 1 boven en karakter 2 onder te plaatsen.

Het percentage bepaalt **het gebied**, niet de exacte grootte van een lichaam.
Beschrijf bijvoorbeeld `a small figure in the background` of
`a large figure in the foreground` als je die compositie wilt.
Houd de gedeelde omgeving en stijl in de base prompt; zet karaktereigenschappen
en triggers in de velden van het betreffende karakter.

## 4. Free boxes: zelf plaatsen en vergroten

Kies `Free boxes` bij Direction. Versleep een kader om het te verplaatsen en
sleep een hoek om het groter of kleiner te maken. Je kunt ook coördinaten typen:

```text
5, 10, 40, 55; 35, 5, 60, 85
```

Elke groep betekent `x, y, breedte, hoogte`, in procenten. `x` wordt vanaf links
gemeten, `y` vanaf boven. De eerste groep hoort bij het blauwe kader 1, de tweede
bij het roze kader 2. Houd `x + breedte` en `y + hoogte` maximaal 100.
Het voorbeeld heeft een kleiner eerste kader links en een groter tweede kader
rechts; ze overlappen gedeeltelijk. De split-slider wordt hier niet gebruikt.

Maak het Boxes-veld leeg om de standaardkaders terug te krijgen. Geldige eigen
coördinaten blijven behouden bij het opslaan en laden van een workflow.

**Overlap:** in het overlappende gebied krijgt de prompt van karakter 1 voorrang.
Dat garandeert niet dat het model karakter 1 letterlijk vóór karakter 2 tekent.
Beschrijf wie op de voorgrond of achtergrond staat in je prompts. Buiten beide
kaders geldt alleen de base prompt. Zorg dat ieder karakter voldoende eigen
gebied overhoudt; als karakter 1 karakter 2 volledig bedekt, stopt de node met
een melding.

**LoRA edge softness:** `0` houdt de LoRA-grenzen hard. Een hogere waarde laat
de invloed van de LoRA’s bij de randen geleidelijk overgaan. De waarde is in
pixels van de gegenereerde afbeelding, niet van de preview. Begin eventueel
bij `32` en verhoog voorzichtig. De prompts blijven gescheiden, maar nabij de
grens kunnen meer uiterlijkdetails van de andere LoRA doorkomen. Deze optie
verandert de kadergrootte of overlapvoorrang niet en bepaalt niet wie vooraan staat.

## 5. Swap characters

De knop wisselt de trigger, prompt, LoRA-keuze en LoRA-sterkte tussen karakter
1 en 2. Bij **Free boxes blijven de kaders op hun plek**. Ieder figuur krijgt
dus de locatie en grootte van het andere kader.

Voorbeeld: blauw 1 is een vrouw en roze 2 een man. Na Swap staat de man in blauw
1 en de vrouw in roze 2. De nummers en kleuren horen bij de gebieden, niet bij
een vaste identiteit. De voorrang bij overlap blijft bij gebied 1.

Bij **links/rechts of boven/onder** wisselt ook het percentage: 60/40 wordt
40/60. Zo behoudt ieder figuur zijn aandeel terwijl het van kant wisselt.
De knop wisselt geen aangesloten masker- of andere inputverbindingen; wissel
die verbindingen handmatig als dat nodig is. Extern aangesloten karakterwaarden
blijven door hun bronnode bepaald.

## 6. Beeldverhouding van de preview

Kies bij **Preview ratio** dezelfde verhouding als bij je resolutie, bijvoorbeeld
`4:5` voor 1152×1440. Dit werkt direct, vóór het genereren.

`Auto` probeert de afmetingen te lezen uit zichtbare width/height-velden,
CyberKrea-resolutielabels en herkenbare latentverbindingen. De optionele ingangen
`width` en `height` kunnen samen op een resolutie-node worden aangesloten.
Waarden die een andere node pas tijdens uitvoering berekent, zijn vooraf niet
altijd te lezen. Gebruik dan een vaste verhouding.

Deze opties bepalen alleen de preview. De **aangesloten LATENT bepaalt altijd
de werkelijke generatiegrootte**. De Composer schaalt die latent niet met de
preview-instelling.

## 7. Maskers en separation

Een MASK op `character_1_mask` of `character_2_mask` vervangt het gebied van dat
karakter, ook als je split of Free boxes hebt gekozen. Het masker wordt aangepast
aan de afmetingen van de afbeelding. Bij een maskerbatch wordt het eerste masker
gebruikt. Wit geeft het karaktergebied aan; te dunne of te grijze gebieden kunnen
op het 16px-raster verdwijnen en geven dan een melding. De kaderpreview toont
geen exacte vorm van een aangesloten masker; bekijk daarvoor `region_masks`.

| Separation | Effect |
|---|---|
| Balanced | Begin hiermee. Karakterprompts blijven regionaal, terwijl de scène samenhang kan houden. |
| Strong | Beperkt in de eerste helft van sampling ook beeldattention tussen de gebieden. Kan vermenging verminderen, maar ook interactie tussen de figuren beperken. |
| Soft | Experimenteel: in de laatste 40% mogen alle gebieden beide karakterprompts lezen. De LoRA’s blijven regionaal, maar promptdetails kunnen zich mengen. |

**Base strength** is experimenteel. `1.0` is de standaard. Onder 1 wegen de eigen
karakterprompts relatief zwaarder; boven 1 de base prompt. Dit werkt binnen de
karaktergebieden. Probeer bijvoorbeeld 0.5–1.5 als de gedeelde stijl onvoldoende
doorkomt.

## 8. Common Prompts, inline LoRA’s en negatieve prompts

Definieer gedeelde tekst één keer en gebruik de naam in andere velden:

```text
Base prompt:  A classroom, {outfit: navy school uniform, red tie}
Character 1:  a woman with blonde hair, {outfit}
Character 2:  a man with black hair, {outfit}
```

`{outfit: ...}` houdt de tekst op die plek en definieert de verwijzing.
`{outfit}` wordt overal vervangen door die tekst. Een onbekende verwijzing blijft
gewone tekst en wordt gemeld in de `prompts`-uitgang.

Een tag zoals `<lora:my_character:0.8>` in het trigger- of promptveld van een
karakter laadt die LoRA regionaal. Je kunt ook gewoon het LoRA-menu gebruiken.
Als hetzelfde bestand op beide manieren wordt gekozen, wint de menusterkte.
Zet zulke tags niet in de base of negative prompt: voor een globale stijl-LoRA
gebruik je een LoraLoader vóór de Composer. Een daar geladen karakter-LoRA
werkt ook globaal en kan beide figuren beïnvloeden.

Bij Krea 2 Turbo op CFG 1 heeft een normale sampler-negative geen effect.
Met geïnstalleerde NegPiP zet de Composer negatieve tekst om naar negatieve
gewichten in de positive conditioning:

- Heel beeld: vul het Negative prompt-veld, bijvoorbeeld `blurry, watermark`.
- Eén karakter: gebruik een negatief gewicht in diens eigen prompt, bijvoorbeeld
  `a man, (beard:-0.6)`.

Een aparte NegPiP-node is hiervoor niet nodig; de uitbreiding moet wel
geïnstalleerd en beschikbaar blijven voor ComfyUI. Laat de negative-uitgang van
de Composer aangesloten op de sampler: het is een bijpassende lege negative.

## 9. Metadata opslaan

Installeer
[Cyberdelia Image Metadata 2.2.2 of nieuwer](https://github.com/cyberdeliaAI/comfyui-cyberdelia-metadata)
en gebruik **Save Image With Metadata (Cyberdelia)**. Die bewaart de gedeelde
scène, beide karakterprompts inclusief triggers, de oorspronkelijke globale
negative en de regionale LoRA’s met sterktes. Common Prompts zijn dan al ingevuld.

Een extra draad vanaf `prompts` is niet nodig. De informatie blijft ook
beschikbaar als ComfyUI de Composer-uitvoer uit de cache hergebruikt.

## 10. Veelvoorkomende problemen en bijwerken

| Probleem | Oplossing |
|---|---|
| Node verschijnt niet | Controleer de uitgepakte map, herstart ComfyUI en lees de opstartmelding. Werk ComfyUI bij als Krea 2 ontbreekt. |
| Free boxes zijn verdwenen | Maak het Boxes-veld leeg voor standaardkaders, of herstel het formaat met twee groepen van vier getallen. |
| Verkeerde verhouding in de preview | Kies een vaste Preview ratio die bij je latentresolutie past. |
| Figuren mengen | Zet softness op 0, houd karakterdetails uit de base prompt en probeer Strong. Controleer op globaal geladen karakter-LoRA’s. |
| Genereren duurt langer | De gecombineerde prompts, regionale attention en extra LoRA’s vragen meer werk. De optimalisatie slaat LoRA-berekeningen buiten hun maskers over; de totale winst hangt van je hardware en workflow af. |
| NegPiP-melding | Installeer de gelinkte NegPiP-uitbreiding en herstart, of verwijder negatieve prompts/gewichten. |

Bij een ZIP-installatie vervang je de bestaande map door de nieuwe versie.
Bij een Git-installatie voer je `git pull` uit in de Composer-map. Herstart
ComfyUI en vernieuw de browser. Bewaar niet twee geïnstalleerde kopieën naast
elkaar. Oudere 0.2-, 0.3- en 0.4-workflows worden met hun veldwaarden geladen.

Regio’s sturen identiteit en plaatsing, maar garanderen geen perfecte scheiding.
Controleer je resultaat vooral bij overlap, interactie en handen over de grens.
