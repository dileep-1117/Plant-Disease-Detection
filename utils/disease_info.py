"""
Agronomic Disease Knowledge Base
=================================
Complete pathology catalog covering all 38 PlantVillage classes:
- Symptoms
- Pathogens & Causes
- Preventative Cultivation Practices
- Recommended Curative Treatments & Management
"""

from typing import Any, Dict, List, Optional

DISEASE_CATALOG: Dict[str, Dict[str, Any]] = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "is_healthy": False,
        "pathogen": "Venturia inaequalis (Ascomycete Fungus)",
        "summary": "Apple scab is a severe fungal disease affecting apple trees (Malus domestica), resulting in defoliation, olive-brown to black velvety lesions on leaves, and corky, scabby lesions on fruit that reduce yield and marketability.",
        "symptoms": [
            "Olive-green to velvety brown or black circular spots on leaves.",
            "Leaf distortion, yellowing around lesions, and premature leaf drop.",
            "Cracked, corky, olive-colored scab lesions on fruit skin.",
            "Deformed, stunted fruit growth leading to premature fruit drop.",
        ],
        "causes": [
            "Overwinters in fallen infected leaves on the orchard floor.",
            "Ascospores released during spring rain events.",
            "Prolonged leaf wetness (6+ hours) at temperatures between 15°C and 24°C.",
            "Poor air circulation within dense tree canopies.",
        ],
        "prevention": [
            "Plant resistant apple cultivars (e.g., Liberty, Enterprise, Freedom, Pristine).",
            "Rake, mulch, or destroy fallen leaves in autumn to disrupt the primary inoculum cycle.",
            "Prune branches annually in dormant season to promote sunlight penetration and air circulation.",
            "Utilize drip or under-canopy irrigation rather than overhead sprinklers.",
        ],
        "treatments": [
            "Apply protective fungicides (captan, mancozeb) starting at green tip stage before rain.",
            "Apply systemic fungicides (myclobutanil, difenoconazole) if primary infection is detected.",
            "Organic orchards: Apply sulfur, lime-sulfur, or copper soaps early in the growing season.",
            "Ensure thorough spray coverage on both upper and lower leaf surfaces.",
        ],
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot (Frogeye Leaf Spot)",
        "is_healthy": False,
        "pathogen": "Botryosphaeria obtusa (Fungus)",
        "summary": "Black rot causes frogeye leaf spots on foliage, sunken cankers on limbs, and firm, dark rot on maturing fruit, frequently leading to limb dieback and severe post-harvest crop loss.",
        "symptoms": [
            "Small purple spots on leaves expanding into circular lesions with tan centers and purple margins (frogeye).",
            "Reddish-brown firm fruit decay that turns completely black and mummifies on the branch.",
            "Depressed reddish-brown cankers on bark, girdling small twigs and limbs.",
            "Black pycnidia (tiny fungal bumps) visible on mummified fruit and bark cankers.",
        ],
        "causes": [
            "Fungus overwinters in dead wood, pruning stubs, fire blight cankers, and mummified fruit.",
            "Spores spread by splashing rain and wind currents in warm weather (20°C - 27°C).",
            "Infection enters fruit and foliage through mechanical wounds or insect punctures.",
        ],
        "prevention": [
            "Prune out dead wood, cankered limbs, and fire blight strikes at least 15 cm below visible margins.",
            "Remove and destroy all mummified fruit remaining on the tree or orchard floor.",
            "Control codling moth and other insects to avoid skin punctures.",
            "Burn or bury all pruned infected material away from the orchard.",
        ],
        "treatments": [
            "Apply captan, thiophanate-methyl, or strobilurin fungicides from silver tip through harvest.",
            "Apply copper sprays at late dormant stage before bud break.",
            "Maintain balanced tree nutrition and avoid excessive nitrogen application.",
        ],
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "is_healthy": False,
        "pathogen": "Gymnosporangium juniperi-virginianae (Heteroecious Rust Fungus)",
        "summary": "Cedar apple rust is a two-host fungal disease requiring both apple/crabapple trees and eastern red cedar (Juniperus virginiana) to complete its complex multi-spore life cycle.",
        "symptoms": [
            "Bright orange-yellow circular spots on the upper leaf surface appearing in late spring.",
            "Spots enlarge with tiny black fruiting bodies (pycnia) in the center.",
            "Cylindrical spore cups (aecia) with fringe-like borders develop on the leaf underside.",
            "Severe infections cause defoliation and small, distorted, rusted fruit.",
        ],
        "causes": [
            "Windblown aeciospores travel from juniper galls (cedar apples) during wet spring periods.",
            "Warm spring temperatures (13°C - 24°C) with persistent leaf wetness.",
            "Proximity of eastern red cedar trees within 1 to 2 miles of apple orchards.",
        ],
        "prevention": [
            "Plant rust-resistant cultivars (e.g., Redfree, Liberty, William's Pride, Enterprise).",
            "Eradicate nearby red cedar trees or hand-prune galls from ornamental junipers before spring rains.",
            "Establish windbreak barriers between juniper woodlands and the orchard.",
        ],
        "treatments": [
            "Apply preventative fungicides (myclobutanil, propiconazole, trifloxystrobin) starting at pink bud through petal fall.",
            "Ensure applications precede rainy forecast periods when orange gelatinous galls release spores.",
        ],
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": "None - Plant Foliage Healthy",
        "summary": "Apple foliage shows vibrant green coloration, uniform leaf lamina, clear venation, intact leaf margins, and complete absence of fungal spots, rust pustules, or insect galls.",
        "symptoms": [
            "Crisp, uniform green leaves with serrated margins.",
            "Intact cuticle layer without chlorosis, necrosis, or lesions.",
            "Strong petiole attachment and healthy shoot elongation.",
        ],
        "causes": [
            "Optimal orchard management, balanced nutrient supply, and adequate moisture.",
        ],
        "prevention": [
            "Maintain annual dormant pruning and balanced N-P-K fertilization.",
            "Monitor periodically for early signs of codling moth, aphids, and mites.",
            "Ensure regular irrigation schedule and soil health maintenance.",
        ],
        "treatments": [
            "No disease treatment required. Continue routine orchard care and preventive pest scouting.",
        ],
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": "None - Plant Foliage Healthy",
        "summary": "Blueberry bush leaves display deep, glossy green foliage, robust vigor, healthy acidic soil response, and absence of rust spots, anthracnose, or stem blight.",
        "symptoms": [
            "Glossy, smooth dark green leaves with uniform surface.",
            "Robust shoot growth without interveinal yellowing or necrotic edges.",
            "Healthy bud clusters and vigorous branch development.",
        ],
        "causes": [
            "Acidic soil conditions (pH 4.5 - 5.5), good organic matter, and regular drip irrigation.",
        ],
        "prevention": [
            "Maintain soil pH between 4.5 and 5.5 with elemental sulfur or acidic fertilizers.",
            "Apply pine bark or sawdust mulch to preserve moisture and suppress weeds.",
            "Prune older canes every winter to maintain 6-8 vigorous main stems.",
        ],
        "treatments": [
            "No chemical treatment required. Sustain regular acidic irrigation and seasonal mulch.",
        ],
    },
    "Cherry___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": "None - Plant Foliage Healthy",
        "summary": "Cherry leaves exhibit rich dark green color, healthy serrated margins, clean gloss, and complete absence of powdery mildew, shot-hole perforations, or black knot.",
        "symptoms": [
            "Uniform green leaf blade with prominent central vein and smooth margins.",
            "No powdery fungal growth, leaf spots, or shot-hole perforations.",
            "Sturdy petiole with distinct pair of healthy glands near leaf base.",
        ],
        "causes": [
            "Good air circulation, proper dormant spray routine, and optimal tree vigor.",
        ],
        "prevention": [
            "Prune during dry summer periods to reduce bacterial canker and silver leaf risks.",
            "Apply dormant copper or horticultural oils prior to bud swell.",
            "Maintain weed-free root zone and deep, well-draining soil.",
        ],
        "treatments": [
            "No curative treatment needed. Continue good orchard sanitation and balanced watering.",
        ],
    },
    "Cherry___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "is_healthy": False,
        "pathogen": "Podosphaera clandestina (Erysiphaceae Fungus)",
        "summary": "Cherry powdery mildew is a foliar fungal infection producing white, powdery fungal mycelia and conidia on leaves, shoots, and developing fruit, leading to leaf curling, stunting, and damaged fruit skins.",
        "symptoms": [
            "Circular patches of white, powdery mildew on young leaves and shoots.",
            "Leaf margins curling upward, distorting, and becoming brittle.",
            "Severe infection leads to premature defoliation and blemished fruit skin.",
            "Young terminal shoots become stunted, covered in powdery white spores.",
        ],
        "causes": [
            "Fungus overwinters as chasmothecia in bark crevices and orchard debris.",
            "High relative humidity coupled with warm temperatures (18°C - 28°C).",
            "Dense canopy shading and excessive late-spring nitrogen fertilization.",
        ],
        "prevention": [
            "Prune dense canopies to maximize airflow and direct sunlight penetration.",
            "Avoid excessive nitrogen fertilizer which stimulates susceptible tender shoot flushes.",
            "Use under-tree micro-sprinklers instead of overhead irrigation systems.",
        ],
        "treatments": [
            "Apply sulfur or potassium bicarbonate sprays at shuck fall and repeat at 10-14 day intervals.",
            "Utilize demethylation inhibitor (DMI) or quinone outside inhibitor (QoI) fungicides during heavy disease pressure.",
            "Apply neem oil or potassium salts of fatty acids for organic production.",
        ],
    },
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Cercospora Leaf Spot (Gray Leaf Spot)",
        "is_healthy": False,
        "pathogen": "Cercospora zeae-maydis (Fungus)",
        "summary": "Gray leaf spot is one of the most destructive foliar diseases of maize worldwide, creating elongated rectangular necrotic lesions delimited by leaf veins that severely restrict photosynthetic capacity.",
        "symptoms": [
            "Small, tan, halo-surrounded spots maturing into elongated rectangular gray-to-tan lesions.",
            "Lesions strictly bordered by parallel leaf veins (1-6 cm long, 2-4 mm wide).",
            "Extensive blight across whole canopy during grain fill, causing premature plant lodging.",
            "Stalk rot develops as grain draws carbohydrates from depleted stems.",
        ],
        "causes": [
            "Pathogen survives on corn residue on the soil surface from previous harvest.",
            "Warm, humid weather (temperatures 25°C - 32°C and relative humidity >90%).",
            "Consecutive continuous no-till corn cultivation.",
        ],
        "prevention": [
            "Plant corn hybrids with high genetic tolerance or resistance to Gray Leaf Spot.",
            "Rotate fields out of corn for at least 1-2 years with soybeans, wheat, or alfalfa.",
            "Tillage to bury infected corn residue accelerates decomposition of fungal inoculum.",
        ],
        "treatments": [
            "Foliar fungicide application (strobilurin + triazole premix) between VT (tasseling) and R2 (blister stage).",
            "Scout canopy at V14 stage; treat if lesions appear on third leaf below ear leaf on 50% of plants.",
        ],
    },
    "Corn___Common_rust": {
        "plant": "Corn",
        "disease": "Common Rust",
        "is_healthy": False,
        "pathogen": "Puccinia sorghi (Macrocyclic Rust Fungus)",
        "summary": "Common rust produces raised, powdery, cinnamon-brown pustules (uredinia) on both upper and lower leaf surfaces of corn, breaking open the epidermis to release clouds of rust spores.",
        "symptoms": [
            "Oval to elongate, cinnamon-brown powdery pustules scattered across both leaf surfaces.",
            "Pustules rupture epidermal tissue, turning dark brown to black (teliospores) late in season.",
            "Chlorosis and premature yellowing surrounding dense pustule clusters.",
            "Severe infection causes leaf necrosis and reduces ear fill and bushel weight.",
        ],
        "causes": [
            "Urediniospores blow north from southern regions on storm systems each spring.",
            "Cool to moderate temperatures (16°C - 24°C) with high relative humidity (>95%) and 6+ hours dew.",
            "Late planting exposes young corn to heavy seasonal airborne spore loads.",
        ],
        "prevention": [
            "Select corn hybrids containing specific Rp resistance genes (Rp1-D) or general adult-plant tolerance.",
            "Plant early in the spring to allow crop to advance past critical stages before spores arrive.",
            "Maintain balanced soil fertility, avoiding excessive nitrogen with deficient potassium.",
        ],
        "treatments": [
            "Apply foliar triazole or strobilurin fungicides (e.g., azoxystrobin, propiconazole) if rust reaches 6 or more pustules per leaf prior to tasseling.",
            "Generally not economically justified once crop enters dough (R4) stage.",
        ],
    },
    "Corn___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": "None - Plant Foliage Healthy",
        "summary": "Healthy corn displays deep emerald-green upright leaves with clean midribs, uniform leaf margins, and robust photosynthesis free from fungal rust, blight, or mosaic patterns.",
        "symptoms": [
            "Broad, rich green leaves with prominent white-green central midrib.",
            "Absence of pustules, streaks, chlorosis, or necrotic margins.",
            "Strong stalk strength and uniform tassel/ear development.",
        ],
        "causes": [
            "Adequate nitrogen, phosphorus, and zinc fertility, balanced rainfall, and good hybrid vigor.",
        ],
        "prevention": [
            "Maintain soil testing, proper seeding density, and balanced side-dress nitrogen.",
            "Perform scouting for armyworms, corn borer, and rootworm pressure.",
            "Practice multi-year crop rotation with non-host legumes.",
        ],
        "treatments": [
            "No chemical treatment required. Sustain standard crop management and nutrient programs.",
        ],
    },
    "Corn___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Corn Leaf Blight (NCLB)",
        "is_healthy": False,
        "pathogen": "Exserohilum turcicum (Fungus)",
        "summary": "Northern Corn Leaf Blight is a serious foliar disease producing large, cigar-shaped grayish-green to tan lesions that can quickly cause massive defoliation and yield reduction.",
        "symptoms": [
            "Distinctive long, elliptical, cigar-shaped lesions (2.5 to 15 cm long).",
            "Initial lesions are grayish-green, turning tan to brown with age.",
            "Dark olive-green fuzzy fungal sporulation inside lesions under damp weather.",
            "Extensive blighting coalesces, killing complete leaf blades before silking.",
        ],
        "causes": [
            "Overwinters in infected corn debris from the previous harvest.",
            "Moderate temperatures (18°C - 27°C) accompanied by extended periods of dew (6-18 hours).",
            "Monoculture corn cropping combined with conservation minimum tillage.",
        ],
        "prevention": [
            "Plant corn hybrids with Ht resistance genes (Ht1, Ht2, Ht3, or HtN) or strong multigenic tolerance.",
            "Rotate with non-host crops like soybeans for at least one full season.",
            "Incorporate corn residue thoroughly with tillage to accelerate fungal decay.",
        ],
        "treatments": [
            "Apply dual-mode foliar fungicides (strobilurin + triazole, e.g., pyraclostrobin + metconazole).",
            "Optimal application timing is just before tassel emergence (VT) up to silk emergence (R1).",
        ],
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "is_healthy": False,
        "pathogen": "Guignardia bidwellii (Fungus)",
        "summary": "Grape black rot is a devastating fungal pathogen that attacks leaves, shoots, canes, and berries, converting healthy grape bunches into shriveled, hard, black mummies.",
        "symptoms": [
            "Small yellowish circular leaf spots expanding with dark brown margins and tiny black dots.",
            "Sunken, purple-to-black elliptical cankers on shoots and tendrils.",
            "Berries develop soft brown decay that rapidly dries into hard, wrinkled black mummies.",
            "Pycnidia visible as tiny black pimples on the surface of mummified grapes.",
        ],
        "causes": [
            "Overwinters in mummified berries hanging on vines or lying on vineyard ground.",
            "Ascospores and conidia discharged during rainy periods in warm weather (21°C - 32°C).",
            "Leaf and berry wetness lasting 6 to 24 hours.",
        ],
        "prevention": [
            "Sanitation: Collect, remove, and destroy all mummified grape clusters during winter pruning.",
            "Canopy management: Shoot thinning and leaf pulling around fruit clusters to improve air movement.",
            "Choose less susceptible grape varieties where suitable for the viticultural zone.",
        ],
        "treatments": [
            "Fungicide sprays essential from early bloom through 4-6 weeks after bloom.",
            "Use myclobutanil, tebuconazole, mancozeb, or kresoxim-methyl according to spray schedule.",
            "Organic vineyards: Copper hydroxide and sulfur combinations applied early in season.",
        ],
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles / Grapevine Trunk Disease)",
        "is_healthy": False,
        "pathogen": "Complex of fungi (Phaeomoniella chlamydospora, Phaeoacremonium minimum, Fomitiporia mediterranea)",
        "summary": "Esca is a complex destructive trunk disease attacking the vascular system of grapevines, characterized by tiger-stripe leaf patterns, spotted fruit (black measles), and internal wood rot.",
        "symptoms": [
            "Interveinal chlorosis and necrosis creating a distinctive 'tiger-stripe' pattern on leaves.",
            "Berries develop dark purple or black speckled spots ('black measles') and crack.",
            "Internal cross-sections of trunk reveal dark vascular streaking and spongy yellow decay.",
            "Apoplexy: Sudden midsummer collapse and complete wilting of entire vines within days.",
        ],
        "causes": [
            "Fungal spores enter vines through winter pruning wounds.",
            "Spread by wind and rain splashing; slow progression over several years.",
            "Heat and water stress in midsummer trigger rapid symptom expression.",
        ],
        "prevention": [
            "Delay pruning until late winter when wound healing is faster.",
            "Apply wound sealant (pruning paint) or bio-fungicide paste (Trichoderma) to pruning cuts.",
            "Remove and incinerate dead and dying infected vines from the vineyard.",
            "Avoid aggressive mechanical wounds to vine trunks.",
        ],
        "treatments": [
            "No chemical cure exists for infected internal heartwood.",
            "Trunk renewal: Cut back trunk below diseased wood to train a healthy suckered cane.",
            "Remedial surgery to excise spongy decayed wood can extend productive vine life.",
        ],
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": "None - Plant Foliage Healthy",
        "summary": "Grapevine canopy displays uniform vibrant green, smooth palmate leaves with sharp lobing, vigorous tendril growth, and clean clusters free from downy mildew, powdery mildew, or rot.",
        "symptoms": [
            "Deep green leaves with crisp margins and clear, unobstructed leaf veins.",
            "Clean leaf undersides without downy white felt or powdery patches.",
            "Vigorous shoot growth and strong fruit set.",
        ],
        "causes": [
            "Balanced canopy management, adequate trellis ventilation, and proactive vineyard management.",
        ],
        "prevention": [
            "Maintain trellis training, shoot positioning, and timely dormant pruning.",
            "Ensure good vineyard drainage and avoid overhead vine wetting.",
            "Scout for grape berry moth and leafhoppers.",
        ],
        "treatments": [
            "No disease treatments required. Maintain standard organic or conventional vineyard hygiene.",
        ],
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight (Isariopsis Leaf Spot / Phaeoisariopsis)",
        "is_healthy": False,
        "pathogen": "Phaeoisariopsis clavispora / Pseudocercospora vitis (Fungus)",
        "summary": "Isariopsis leaf blight causes irregular reddish-brown necrotic spots on mature leaves, leading to severe premature defoliation late in the season and reduced vine winter hardiness.",
        "symptoms": [
            "Irregular, angular, dark reddish-brown to black spots (2-8 mm) on leaves.",
            "Lesions bordered by a distinct yellowish or dark reddish halo.",
            "Olive-brown felt-like fungal sporulation visible on the lesion undersides in humid air.",
            "Heavy infection causes leaves to dry, curl, and drop prematurely before grape harvest.",
        ],
        "causes": [
            "Pathogen overwinters in fallen leaves and dormant cane bark.",
            "Warm, wet, and humid conditions in mid-to-late summer.",
            "Poor air movement in overgrown or unpruned canopies.",
        ],
        "prevention": [
            "Perform leaf thinning around the fruit zone to improve wind flow.",
            "Rake or disc under fallen grape leaves after leaf drop in autumn.",
            "Avoid excessive overhead irrigation.",
        ],
        "treatments": [
            "Apply broad-spectrum copper fungicides or mancozeb during the growing season.",
            "Strobilurin or azole fungicides applied for downy/powdery mildew also control leaf blight.",
        ],
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing (Citrus Greening)",
        "is_healthy": False,
        "pathogen": "Candidatus Liberibacter asiaticus (Fastidious Phloem-Limited Bacterium)",
        "summary": "Huanglongbing (HLB / Yellow Dragon Disease) is the most destructive citrus disease globally, causing asymmetric yellow leaf mottling, phloem blockage, twig dieback, and bitter, lopsided fruit.",
        "symptoms": [
            "Blotchy, asymmetric chlorotic mottle across the leaf blade (vein crossing yellowing).",
            "Yellowing of individual shoots, creating a 'yellow shoot' appearance in the green canopy.",
            "Thickened, corky, or raised leaf veins and upright, leathery leaves.",
            "Small, lopsided, poorly colored fruit with bitter juice and aborted dark seeds.",
        ],
        "causes": [
            "Vector transmission: Asian Citrus Psyllid (Diaphorina citri) feeds on phloem sap.",
            "Graft transmission from infected budwood.",
            "Bacterium multiplies in the sieve tubes, blocking nutrient transport.",
        ],
        "prevention": [
            "Plant only certified disease-free citrus nursery stock.",
            "Control Asian Citrus Psyllid populations aggressively using systemic and contact insecticides.",
            "Inspect trees frequently; promptly identify and remove HLB-positive trees.",
            "Quarantine enforcement: Do not move citrus plants or uncertified cuttings outside regulated zones.",
        ],
        "treatments": [
            "No known cure for infected trees; infected trees decline and must be eradicated.",
            "Nutritional foliar sprays (zinc, manganese, iron, potassium) can temporarily mitigate decline in commercial blocks.",
            "Release biological control agents (Tamarixia radiata wasp) to suppress psyllids.",
        ],
    },
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "is_healthy": False,
        "pathogen": "Xanthomonas arboricola pv. pruni (Gram-Negative Bacterium)",
        "summary": "Bacterial spot attacks leaves, twigs, and fruit of peaches, causing angular purple-brown spots that fall out to leave a 'shot-hole' appearance, followed by twig cankers and pitted fruit.",
        "symptoms": [
            "Small water-soaked polygonal angular spots on leaves, turning deep purple-brown.",
            "Centers of leaf spots drop out, producing a characteristic shot-hole perforated appearance.",
            "Leaf yellowing starting at the leaf tips, accompanied by heavy premature defoliation.",
            "Fruit exhibits pitted, sunken black spots, gum exudation, and deep skin cracking.",
        ],
        "causes": [
            "Bacteria overwinter in spring cankers on young twigs and in dormant buds.",
            "Splashed by rain and driven by wind during warm, humid storms (20°C - 30°C).",
            "Sandy soils, windblown sand abrasions, and weakened tree vigor increase susceptibility.",
        ],
        "prevention": [
            "Select resistant peach varieties (e.g., Clayton, Candor, PF-1, Reliance).",
            "Avoid planting orchards in highly wind-exposed sandy ridges without windbreaks.",
            "Maintain optimal tree nutrition; avoid high nitrogen flushes in late summer.",
        ],
        "treatments": [
            "Apply dormant and delayed-dormant copper sprays before bud swell.",
            "Apply low-dose oxytetracycline or copper sprays at 7-10 day intervals from petal fall to shuck split.",
            "Exercise caution with copper during foliage periods to avoid phytotoxic leaf burning.",
        ],
    },
    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": "None - Plant Foliage Healthy",
        "summary": "Peach foliage is vibrant lanceolate green with clean margins, glossy upper surfaces, sturdy petioles, and completely free from bacterial shot-holes, leaf curl, or cankers.",
        "symptoms": [
            "Bright, glossy green lanceolate leaves with finely serrate margins.",
            "No curl, red blistering, shot holes, or gummy twig exudate.",
            "Robust shoot growth with clean, healthy peach fruitlets.",
        ],
        "causes": [
            "Timely dormant copper application, balanced irrigation, and good canopy training.",
        ],
        "prevention": [
            "Apply dormant spray of copper or chlorothalonil for peach leaf curl prevention.",
            "Prune into open-center vase structure to promote full interior sunlight.",
            "Thin fruit early in the season to maintain vigor and prevent branch breakage.",
        ],
        "treatments": [
            "No treatment necessary. Continue standard seasonal stone-fruit management.",
        ],
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Pepper (Bell)",
        "disease": "Bacterial Spot",
        "is_healthy": False,
        "pathogen": "Xanthomonas euvesicatoria / Xanthomonas campestris pv. vesicatoria",
        "summary": "Bacterial spot is a devastating foliar and fruit disease of bell peppers, causing water-soaked spots that become necrotic, causing severe defoliation and blistered, sunscalded fruit.",
        "symptoms": [
            "Small, circular-to-irregular water-soaked spots on leaf undersides, turning dark brown.",
            "Spots become sunken on upper surface with a slight yellow halo.",
            "Widespread leaf yellowing and rapid defoliation, exposing fruit to sunscald.",
            "Raised, rough, scab-like circular blisters on bell pepper fruit skin.",
        ],
        "causes": [
            "Seedborne bacterial infection and survival in infected crop debris.",
            "Warm temperatures (24°C - 30°C) with frequent rainfall, overhead irrigation, or heavy dew.",
            "Bacteria enter leaves through stomata and microscopic abrasion wounds.",
        ],
        "prevention": [
            "Use certified pathogen-free pepper seed; hot water seed treatment (50°C for 25 min).",
            "Plant resistant pepper cultivars bred with Xanthomonas resistance races.",
            "Avoid overhead irrigation; use drip irrigation underneath plastic mulch.",
            "Rotate crops away from solanaceous plants (tomatoes, eggplants) for at least 2 years.",
        ],
        "treatments": [
            "Apply preventative copper bactericide combined with mancozeb for improved efficacy.",
            "Apply plant defense activators such as acibenzolar-S-methyl (Actigard).",
            "Apply bacteriophages specific to Xanthomonas strains where commercially available.",
        ],
    },
    "Pepper,_bell___healthy": {
        "plant": "Pepper (Bell)",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": "None - Plant Foliage Healthy",
        "summary": "Bell pepper plant exhibits broad, glossy, dark green leaves with smooth margins, strong stems, healthy white blossom flowers, and absence of bacterial spots or viruses.",
        "symptoms": [
            "Smooth, deep green leaves without mottling, curling, or brown spots.",
            "Strong main stem and healthy branching structure.",
            "Normal blossom set and firm, glossy green bell peppers.",
        ],
        "causes": [
            "Adequate warmth, well-draining soil, regular drip moisture, and balanced calcium/magnesium.",
        ],
        "prevention": [
            "Maintain consistent drip watering to avoid blossom-end rot.",
            "Mulch beds to regulate soil temperature and suppress weeds.",
            "Scout weekly for aphids, thrips, and hornworms.",
        ],
        "treatments": [
            "No disease treatment required. Continue standard vegetative and flowering care.",
        ],
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "is_healthy": False,
        "pathogen": "Alternaria solani (Fungus)",
        "summary": "Early blight is a common foliar fungal disease of potato and tomato, characterized by brown-black spots with concentric rings ('target-board' pattern) primarily on mature lower leaves.",
        "symptoms": [
            "Dark brown to black circular-to-angular spots on older lower leaves.",
            "Concentric rings inside lesions producing a distinctive 'target board' or bullseye appearance.",
            "Yellowing (chlorosis) of leaf tissue surrounding the spots.",
            "Premature senescence of lower canopy leaves, reducing tuber size and dry matter.",
        ],
        "causes": [
            "Fungus overwinters in plant debris, soil, or infected seed tubers.",
            "Alternating periods of wet and dry weather with temperatures of 20°C - 30°C.",
            "Plant stress from high nitrogen demand, heavy tuber bulking, or drought.",
        ],
        "prevention": [
            "Use certified disease-free seed tubers.",
            "Practice 3-to-4 year crop rotation with non-solanaceous crops (grasses, grains, legumes).",
            "Maintain optimal plant nutrition and avoid drought stress during tuber bulking.",
            "Avoid overhead sprinkler irrigation late in the afternoon or evening.",
        ],
        "treatments": [
            "Apply protectant fungicides (mancozeb, chlorothalonil) starting before canopy closure.",
            "Rotate with systemic fungicides (azoxystrobin, difenoconazole, boscalid) during active disease pressure.",
            "Organic growers: Apply copper sulfate or Bacillus subtilis formulations.",
        ],
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": "None - Plant Foliage Healthy",
        "summary": "Healthy potato plants feature lush, compound emerald-green foliage with smooth leaf margins, sturdy stems, upright habit, and complete freedom from blight, mosaic, or leaf roll.",
        "symptoms": [
            "Uniform compound green leaves with crisp edges.",
            "No target spots, water-soaked margins, or curled edges.",
            "Vigorous vine growth and healthy flowering/tuber set.",
        ],
        "causes": [
            "Certified seed tubers, well-hilled soil beds, balanced N-P-K, and consistent moisture.",
        ],
        "prevention": [
            "Ensure good soil hilling to protect developing tubers from sunlight and blight spores.",
            "Monitor fields regularly for Colorado potato beetle and potato leafhoppers.",
            "Destroy volunteer potato plants in adjacent fields.",
        ],
        "treatments": [
            "No chemical disease treatment needed. Maintain weed control and balanced irrigation.",
        ],
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "is_healthy": False,
        "pathogen": "Phytophthora infestans (Oomycete / Water Mold)",
        "summary": "Late blight is the infamous pathogen that caused the Irish Potato Famine. It is an extremely destructive water mold capable of destroying entire potato fields within a few days under cool, wet conditions.",
        "symptoms": [
            "Water-soaked, pale to dark green irregular lesions turning brown-black rapidly.",
            "White, velvety fungal-like downy growth visible on lesion undersides in high humidity.",
            "Leaves, petioles, and stems rapidly collapse into a smelly, dark decaying mass.",
            "Infected tubers exhibit copper-brown granular rot beneath the skin extending into the flesh.",
        ],
        "causes": [
            "Overwinters in infected cull piles, volunteer potato tubers, and infected seed potatoes.",
            "Cool, wet conditions: temperatures of 12°C - 22°C and relative humidity >90%.",
            "Sporangia spread by wind and rain splashing over several miles.",
        ],
        "prevention": [
            "Destroy all potato cull piles, compost heaps, and volunteer potato plants completely.",
            "Plant only certified disease-free seed tubers.",
            "Plant resistant or tolerant varieties (e.g., Sarpo Mira, Defender, Elba).",
            "Monitor regional blight forecasting and alert networks.",
        ],
        "treatments": [
            "Apply protectant fungicides (chlorothalonil, mancozeb, fluazinam) preventatively.",
            "Apply translaminar/systemic oomycete fungicides (cymoxanil, mandipropamid, dimethomorph).",
            "If late blight strikes near harvest: Desiccate vines chemically to prevent tuber infection.",
        ],
    },
    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": "None - Plant Foliage Healthy",
        "summary": "Raspberry canes show healthy trifoliate-to-pinnate leaves with silvery undersides, strong upright canes, bright blossoms, and no signs of cane blight, rust, or anthracnose.",
        "symptoms": [
            "Crisp green leaves with distinctive silver-green underside.",
            "Firm, upright primocanes and floricanes with no bark lesions or cracking.",
            "Normal floral development and clean berry drupelets.",
        ],
        "causes": [
            "Well-drained soil, annual cane thinning, and trellised growth.",
        ],
        "prevention": [
            "Prune out spent floricanes immediately after fruiting to improve airflow.",
            "Ensure raised beds to prevent Phytophthora root rot.",
            "Keep rows narrow (45-60 cm) to allow sunlight throughout the lower canes.",
        ],
        "treatments": [
            "No treatment needed. Continue routine cane renewal pruning and seasonal mulching.",
        ],
    },
    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": "None - Plant Foliage Healthy",
        "summary": "Soybean plants display vibrant, clean trifoliate leaves with healthy nodulation on roots, strong canopy closure, and absence of frogeye leaf spot, rust, or sudden death syndrome.",
        "symptoms": [
            "Uniform dark green trifoliate leaves with smooth margins.",
            "Absence of interveinal chlorosis, rust pustules, or necrotic lesions.",
            "Healthy pod formation and vigorous branching.",
        ],
        "causes": [
            "Effective rhizobial nodulation, balanced soil pH (6.0 - 6.8), and crop rotation.",
        ],
        "prevention": [
            "Plant certified soybean seeds treated with appropriate fungicide/inoculant.",
            "Rotate annually with non-host crops (corn, wheat).",
            "Maintain weed management to eliminate competitive weeds and alternative pest hosts.",
        ],
        "treatments": [
            "No disease treatment required. Continue standard field scouting and weed control.",
        ],
    },
    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "is_healthy": False,
        "pathogen": "Podosphaera xanthii / Golovinomyces cichoracearum (Fungi)",
        "summary": "Powdery mildew is a ubiquitous fungal infection on cucurbits, producing talcum powder-like white fungal colonies across leaf surfaces, leading to early defoliation, sunburned fruit, and poor fruit sugar.",
        "symptoms": [
            "White, talcum powder-like circular fungal spots on upper and lower leaf surfaces.",
            "Spots rapidly expand and merge to cover entire leaves and petioles.",
            "Infected leaves turn yellow, then brown, dry up like parchment, and die prematurely.",
            "Loss of canopy cover leads to fruit sunscald, reduced yields, and poor taste.",
        ],
        "causes": [
            "Airborne conidia travel long distances on air currents.",
            "Warm, dry days with high relative humidity at night (20°C - 27°C).",
            "Dense plant canopies with shaded lower foliage.",
        ],
        "prevention": [
            "Plant resistant or tolerant squash varieties (e.g., Success PM, Dunja).",
            "Provide ample spacing between hills and rows to maximize sunlight and airflow.",
            "Avoid overhead watering; irrigate at soil level with drip lines.",
        ],
        "treatments": [
            "Apply potassium bicarbonate, horticultural oils, or neem oil at first sign of white spots.",
            "Apply conventional fungicides (quinoxyfen, myclobutanil, cyflufenamid) in rotation.",
            "Ensure complete coverage of lower leaf surfaces and inner canopy leaves.",
        ],
    },
    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": "None - Plant Foliage Healthy",
        "summary": "Strawberry plants exhibit robust dark green trifoliate leaves with prominent serrations, healthy white flower petals, vigorous runner stolons, and complete absence of leaf scorch or spots.",
        "symptoms": [
            "Deep green, glossy trifoliate leaves with intact serrated edges.",
            "No purple spotting, marginal scorch, or powdery fungal residue.",
            "Healthy crown structure and clean, vigorous runner production.",
        ],
        "causes": [
            "Well-drained sandy loam soil, drip irrigation under straw or plastic mulch, and clean planting stock.",
        ],
        "prevention": [
            "Plant certified disease-free strawberry runners.",
            "Mulch with clean straw to keep leaves and ripening berries off bare soil.",
            "Renovate June-bearing beds annually after harvest.",
        ],
        "treatments": [
            "No treatment needed. Continue good weed control and consistent soil moisture.",
        ],
    },
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "is_healthy": False,
        "pathogen": "Diplocarpon earlianum (Ascomycete Fungus)",
        "summary": "Strawberry leaf scorch causes numerous small, dark purple irregular blotches on leaves that enlarge and coalesce, causing the leaf tissue between spots to turn brown and scorched.",
        "symptoms": [
            "Numerous small, irregular purple to dark brown spots on upper leaf surfaces.",
            "Spots remain purple or brownish throughout without white centers (distinguishing from leaf spot).",
            "Tissues between spots turn purple, brown, and dry out, giving leaves a burned/scorched look.",
            "In severe cases, entire leaves curl upward, wither, and die.",
        ],
        "causes": [
            "Fungus overwinters on living infected leaves and plant crowns.",
            "Conidia spread by rain splash and overhead irrigation during spring and autumn.",
            "Frequent leaf wetness (8-12 hours) and warm temperatures (18°C - 25°C).",
        ],
        "prevention": [
            "Plant resistant strawberry cultivars (e.g., Allstar, Earliglow, Honeoye, Tribute).",
            "Plant in well-drained soil with full sun exposure and wide spacing.",
            "Mow and renovate beds after harvest to destroy old diseased foliage.",
            "Use drip irrigation rather than overhead sprinklers to keep foliage dry.",
        ],
        "treatments": [
            "Apply protectant fungicides (captan, thiram) starting in early spring as new leaves emerge.",
            "Apply systemic fungicides (myclobutanil, azoxystrobin) if disease was severe previous season.",
        ],
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "is_healthy": False,
        "pathogen": "Xanthomonas perforans / Xanthomonas vesicatoria (Bacteria)",
        "summary": "Bacterial spot of tomato causes dark, water-soaked circular lesions on leaves, stems, and fruit, causing severe defoliation, blossom drop, and unmarketable scabby fruit.",
        "symptoms": [
            "Small, dark, water-soaked circular spots (<3 mm) on leaves, becoming black and angular.",
            "Lesions surrounded by a prominent chlorotic yellow halo.",
            "Centers of older lesions may dry, crack, and tear.",
            "Raised, scab-like, blistered dark spots with translucent margins on green fruit.",
        ],
        "causes": [
            "Infected seed or infected seedling transplants.",
            "Warm temperatures (24°C - 30°C) with persistent rain, storms, or overhead sprinkler irrigation.",
            "Bacteria enter via stomata and mechanical wounds from handling or windblown dust.",
        ],
        "prevention": [
            "Use certified pathogen-free seeds and transplants; hot water treat seed.",
            "Rotate out of solanaceous crops for a minimum of 2-3 years.",
            "Use drip irrigation exclusively and avoid working in fields when foliage is wet.",
            "Disinfect stakes, cages, and pruning shears between plants with 10% bleach solution.",
        ],
        "treatments": [
            "Apply fixed copper bactericides tank-mixed with mancozeb to increase soluble copper activity.",
            "Apply plant elicitors (acibenzolar-S-methyl) early in seedling development.",
            "Utilize bacteriophage sprays (OmniLytics AgriPhage) in evening hours for biological control.",
        ],
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "is_healthy": False,
        "pathogen": "Alternaria linariae (formerly Alternaria solani)",
        "summary": "Early blight is a widespread fungal disease attacking tomato foliage, stems, and fruit, producing characteristic dark brown lesions with concentric rings (bullseye/target pattern).",
        "symptoms": [
            "Dark brown to black spots with concentric rings ('target pattern') on older lower leaves.",
            "Yellow halo develops around lesions, followed by total leaf yellowing and drop.",
            "Dark, sunken, leathery cankers on stems near soil line (collar rot).",
            "Dark, sunken, leathery lesions at the stem end of mature green or ripe tomatoes.",
        ],
        "causes": [
            "Pathogen overwinters in plant debris in soil, solanaceous weeds, or seed.",
            "Warm temperatures (24°C - 29°C) and heavy humidity, dew, or rain.",
            "Stressed plants (heavy fruit load, nitrogen deficiency, drought) are highly susceptible.",
        ],
        "prevention": [
            "Rotate crops away from tomatoes, potatoes, eggplants, and peppers for 3 years.",
            "Stake or cage tomatoes and prune lower leaves 30 cm from ground to prevent soil splash.",
            "Apply organic or plastic mulch around plant bases to create a physical splash barrier.",
            "Ensure regular drip irrigation and balanced nitrogen-potassium fertility.",
        ],
        "treatments": [
            "Apply preventative fungicides (chlorothalonil, mancozeb, copper sulfate) starting early.",
            "Apply systemic fungicides (azoxystrobin, difenoconazole, pyraclostrobin) at first lesion detection.",
            "Organic options: Bio-fungicides (Bacillus subtilis, copper octanoate).",
        ],
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "is_healthy": True,
        "pathogen": "None - Plant Foliage Healthy",
        "summary": "Tomato foliage exhibits lush, deep green compound leaves with healthy glandular hairs, robust stems, vigorous flowering trusses, and complete freedom from blight, mold, or spots.",
        "symptoms": [
            "Rich green pinnate compound leaves with distinct aroma and glandular pubescence.",
            "Clean leaf blades without spots, mottling, curl, or yellow halos.",
            "Vigorous shoot growth, strong fruit clusters, and active blossoms.",
        ],
        "causes": [
            "Proper staking, regular drip irrigation, balanced N-P-K-Ca nutrition, and full sunlight.",
        ],
        "prevention": [
            "Mulch soil around plants and stake securely to keep foliage off ground.",
            "Prune bottom suckers and lower foliage to promote maximum airflow.",
            "Monitor weekly for hornworms, aphids, and whiteflies.",
        ],
        "treatments": [
            "No chemical treatment required. Sustain routine watering and organic fertilizing.",
        ],
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "is_healthy": False,
        "pathogen": "Phytophthora infestans (Oomycete)",
        "summary": "Late blight is a catastrophic disease capable of destroying entire tomato crops in days under cool, wet conditions, producing large water-soaked spots, white downy mold, and greasy fruit rot.",
        "symptoms": [
            "Large, irregular water-soaked spots on leaves that turn greasy dark brown to black.",
            "Delicate white fungal-like sporulation visible on the underside of leaves during humid mornings.",
            "Brown-black lesions on petioles and stems, causing rapid collapse of entire branches.",
            "Golden-brown to dark brown, firm, greasy-looking rot on green or ripe tomato fruit.",
        ],
        "causes": [
            "Survives in live plant tissue (infected potato tubers, greenhouse tomato transplants).",
            "Cool (15°C - 22°C), wet, foggy, or rainy weather with relative humidity above 90%.",
            "Windblown sporangia can travel tens of miles on cloud-covered days.",
        ],
        "prevention": [
            "Plant late blight-resistant tomato varieties (e.g., Mountain Magic, Defiant, Plum Regal).",
            "Never plant tomatoes adjacent to potatoes; eliminate all volunteer potato plants.",
            "Ensure wide spacing and high staking to facilitate rapid drying of foliage.",
            "Monitor regional late blight tracking systems (e.g., USAblight).",
        ],
        "treatments": [
            "Apply protectant fungicides (chlorothalonil, mancozeb) before infection occurs.",
            "Apply targeted oomycete fungicides (cymoxanil, mandipropamid, fluopicolide, cyazofamid).",
            "If severe infection is established: Promptly bag and remove infected plants to protect nearby fields.",
        ],
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "is_healthy": False,
        "pathogen": "Passalora fulva (formerly Cladosporium fulvum, Fungus)",
        "summary": "Tomato leaf mold primarily plagues high tunnels and greenhouses with poor ventilation, creating pale green-yellow spots on upper leaf surfaces and velvety olive-brown mold on leaf undersides.",
        "symptoms": [
            "Pale green to yellowish spots with indistinct margins on upper surface of older leaves.",
            "Dense, olive-green to velvety brown mold growth directly underneath the yellow spots.",
            "Leaves turn completely yellow, wither, curl, and drop prematurely.",
            "Blossoms and developing fruit may occasionally be infected and drop.",
        ],
        "causes": [
            "High relative humidity (>85%) with temperatures between 21°C and 24°C.",
            "Poor air circulation inside enclosed greenhouses and high tunnels.",
            "Conidia survive on seed coats, greenhouse framing, and dry plant debris for up to a year.",
        ],
        "prevention": [
            "Ventilate greenhouses and tunnels actively using exhaust fans, ridge vents, and horizontal air fans.",
            "Heat and ventilate simultaneously in evenings to prevent condensation and dew on leaves.",
            "Select tomato cultivars with Cf resistance genes (e.g., Cf-2, Cf-4, Cf-9).",
            "Drip irrigate at ground level; avoid overhead wetting.",
        ],
        "treatments": [
            "Apply copper fungicides, chlorothalonil, or mancozeb at the first detection of yellow spots.",
            "Bio-fungicides: Apply Bacillus amyloliquefaciens or Reynoutria sachalinensis extract.",
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "is_healthy": False,
        "pathogen": "Septoria lycopersici (Fungus)",
        "summary": "Septoria leaf spot is one of the most destructive foliar diseases of tomato, causing numerous small circular spots with gray-white centers and dark borders, resulting in severe bottom-up defoliation.",
        "symptoms": [
            "Numerous small circular spots (1.5 - 3 mm) appearing first on lower leaves after fruit set.",
            "Spots have dark brown to black margins with light gray or tan centers.",
            "Tiny black specks (pycnidia) clearly visible inside the light center of spots with a hand lens.",
            "Severe bottom-up defoliation, leaving plants with bare stems and fruit prone to sunscald.",
        ],
        "causes": [
            "Overwinters in solanaceous crop residue and horsenettle weeds.",
            "Conidia splashed onto lower leaves by rain or overhead irrigation.",
            "Warm temperatures (20°C - 26°C) and extended leaf wetness.",
        ],
        "prevention": [
            "Rotate crops away from tomatoes and solanaceous crops for a minimum of 2-3 years.",
            "Prune lower leaves to a height of 30-40 cm off the soil to break rain-splash trajectory.",
            "Apply heavy mulch (straw, paper, plastic) to seal soil inoculum.",
            "Disinfect tomato stakes, wire cages, and garden tools each season.",
        ],
        "treatments": [
            "Apply protectant fungicides (chlorothalonil, mancozeb, copper) on 7-10 day intervals.",
            "Rotate systemic fungicides (pyraclostrobin, azoxystrobin) to prevent resistance build-up.",
        ],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites (Two-Spotted Spider Mite)",
        "is_healthy": False,
        "pathogen": "Tetranychus urticae (Arachnid / Plant-Feeding Mite Pest)",
        "summary": "Two-spotted spider mites are microscopic sap-sucking pests that pierce leaf cells, causing fine yellow stippling, bronze discoloration, brittle foliage, and extensive webbing covering the canopy.",
        "symptoms": [
            "Fine pale yellow to white stippling (tiny dots) on the upper leaf surface.",
            "Leaves turn chlorotic, bronze, dry, brittle, and drop off.",
            "Fine, silky webbing covering the leaf undersides, growing tips, and fruit clusters.",
            "Tiny moving yellow-green or reddish specks visible on leaf undersides under magnification.",
        ],
        "causes": [
            "Hot, dry, and dusty weather conditions (temperatures >30°C and low humidity).",
            "Overuse of broad-spectrum synthetic pyrethroid insecticides that kill beneficial predatory mites.",
            "Water-stressed plants are especially vulnerable to rapid population explosions.",
        ],
        "prevention": [
            "Keep farm roadways and garden paths moist or mulched to minimize dust.",
            "Ensure plants receive consistent irrigation to prevent drought stress.",
            "Scout leaf undersides regularly, especially along dusty field borders.",
            "Preserve or release natural predatory mites (Phytoseiulus persimilis, Neoseiulus californicus).",
        ],
        "treatments": [
            "Apply insecticidal soap or horticultural mineral oils, ensuring thorough spray coverage on leaf undersides.",
            "Apply neem oil or sulfur sprays (avoid sulfur if temperature exceeds 32°C).",
            "Use selective miticides (bifenazate, abamectin, spiromesifen) if populations exceed threshold.",
        ],
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "is_healthy": False,
        "pathogen": "Corynespora cassiicola (Fungus)",
        "summary": "Target spot produces small brown circular spots that expand into brown lesions with concentric rings, attacking leaves, stems, and causing crater-like lesions on green and red tomatoes.",
        "symptoms": [
            "Small, pinpoint brown spots on leaves with faint yellow halos.",
            "Lesions expand up to 1 cm with noticeable concentric target-like rings.",
            "Extensive foliar blight and defoliation in lower and middle canopies.",
            "Circular, sunken, crater-like brown lesions on green and ripe tomato fruit.",
        ],
        "causes": [
            "Pathogen survives on infected crop debris and numerous alternate weed hosts.",
            "Warm, humid subtropical weather (temperatures 20°C - 30°C with high relative humidity).",
            "Prolonged leaf wetness and poor airflow in dense plantings.",
        ],
        "prevention": [
            "Prune suckers and stake vines properly to open canopy to breezes and light.",
            "Incorporate crop residues deeply immediately after harvest.",
            "Rotate crops for at least 2 seasons with non-hosts like sweet corn or bahia grass.",
            "Avoid overhead irrigation.",
        ],
        "treatments": [
            "Apply broad-spectrum fungicides (chlorothalonil, mancozeb) preventatively.",
            "Apply targeted fungicides (boscalid, pyraclostrobin, fluopyram) at symptom onset.",
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus (ToMV)",
        "is_healthy": False,
        "pathogen": "Tomato Mosaic Tobamovirus (Rigid Rod-Shaped Plant Virus)",
        "summary": "Tomato mosaic virus is an extremely stable, mechanically transmitted virus causing light and dark green mottled mosaic patterns on leaves, leaf distortion (shoestringing), and internal fruit browning.",
        "symptoms": [
            "Mottling of leaves with alternating light and dark green mosaic patterns.",
            "Leaf distortion, blistering, curling, and fern-like or 'shoestring' narrowing.",
            "Overall plant stunting and reduced fruit yield.",
            "Internal brown necrotic rings inside the wall of green or ripening fruit.",
        ],
        "causes": [
            "Highly contagious: Transmitted mechanically via workers' hands, tools, clothing, and pruning shears.",
            "Seedborne on the seed coat.",
            "Extremely stable; survives for years in dry plant debris and uncleaned greenhouse equipment.",
            "Tobacco products can carry the related Tobacco Mosaic Virus (TMV).",
        ],
        "prevention": [
            "Plant ToMV-resistant cultivars (labeled with 'T' or 'ToMV' resistance).",
            "Wash hands thoroughly with soap or milk before handling tomato seedlings.",
            "Ban all tobacco use (smoking, chewing) anywhere near greenhouse or garden areas.",
            "Soak tools in a 20% non-fat dry milk solution or 10% household bleach solution between plants.",
            "Promptly rogue and incinerate infected plants upon detection.",
        ],
        "treatments": [
            "No chemical cure exists for viral plant infections.",
            "Carefully remove and destroy infected plants immediately to prevent mechanical transmission.",
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "is_healthy": False,
        "pathogen": "Begomovirus (Single-Stranded Circular DNA Virus)",
        "summary": "Tomato Yellow Leaf Curl Virus is a devastating viral disease transmitted by the silverleaf whitefly, causing dramatic upward curling of leaves, bright yellowing of leaf margins, severe plant stunting, and near total flower drop.",
        "symptoms": [
            "Marked upward curling (cupping) and crumpling of leaf margins.",
            "Interveinal chlorosis and bright yellowing along leaf margins and young shoots.",
            "Severe plant stunting, giving the plant a compact, erect, bushy appearance.",
            "Flower blossoms drop prematurely, leading to severe yield loss; virtually no fruit set if infected early.",
        ],
        "causes": [
            "Transmitted persistently by the sweetpotato/silverleaf whitefly (Bemisia tabaci).",
            "Not seedborne and not mechanically transmitted by routine pruning or handling.",
            "Hot, dry conditions favorable to large whitefly vector populations.",
        ],
        "prevention": [
            "Plant TYLCV-resistant tomato cultivars (e.g., Tycoon, Charger, Invictus, Foronti).",
            "Install insect-proof fine netting (50-mesh) on greenhouse vents and sidewalls.",
            "Use yellow sticky cards to detect and monitor whitefly populations early.",
            "Use UV-reflective silver or aluminum plastic mulches to repel incoming whiteflies.",
        ],
        "treatments": [
            "No chemical viricide exists to cure infected plants.",
            "Control whitefly vectors using systemic insecticides (neonicotinoids, flupyradifurone, spirotetramat).",
            "Apply insecticidal soap, neem oil, or horticultural oil to smother whitefly nymphs on leaf undersides.",
            "Rogue and bag infected tomato plants immediately to prevent them serving as virus reservoirs.",
        ],
    },
}


def get_disease_details(raw_class_name: str) -> Dict[str, Any]:
    """
    Retrieve full agronomic advisory details for a given raw class name.
    Falls back gracefully if an unexpected class name is requested.
    """
    if raw_class_name in DISEASE_CATALOG:
        return DISEASE_CATALOG[raw_class_name]

    # Fuzzy/normalized match attempt
    normalized = raw_class_name.strip()
    for key, val in DISEASE_CATALOG.items():
        if key.lower() == normalized.lower():
            return val

    # Graceful default fallback
    parts = raw_class_name.split("___")
    plant = parts[0].replace("_", " ").title()
    condition = parts[1].replace("_", " ").title() if len(parts) > 1 else "Condition"
    is_h = "healthy" in raw_class_name.lower()

    return {
        "plant": plant,
        "disease": condition,
        "is_healthy": is_h,
        "pathogen": "Pathogen details pending laboratory analysis",
        "summary": f"Agronomic record for {plant} affected by {condition}.",
        "symptoms": ["Visible foliar discoloration or leaf spotting."],
        "causes": ["Environmental stress, microclimate humidity, or foliar pathogen."],
        "prevention": ["Ensure crop rotation, sanitize tools, and maintain optimal airflow."],
        "treatments": ["Apply standard protective crop management practices and consult local agricultural extension."],
    }


def get_all_plants() -> List[str]:
    """Return a sorted, unique list of all plant species in the catalog."""
    plants = set()
    for info in DISEASE_CATALOG.values():
        plants.add(info["plant"])
    return sorted(list(plants))


def filter_catalog(
    plant_filter: Optional[str] = None,
    search_query: Optional[str] = None,
    status_filter: Optional[str] = "All",
) -> List[Dict[str, Any]]:
    """
    Search and filter the 38-class agronomic disease catalog.
    
    Args:
        plant_filter: Specific plant name or 'All Plants'
        search_query: Substring search across plant, disease, pathogen, or symptoms
        status_filter: 'All', 'Healthy Only', or 'Diseased Only'
    
    Returns:
        List of matching disease record dictionaries.
    """
    results = []
    q = (search_query or "").strip().lower()

    for raw_name, info in DISEASE_CATALOG.items():
        # Plant filter
        if plant_filter and plant_filter != "All Plants" and info["plant"].lower() != plant_filter.lower():
            continue

        # Status filter
        if status_filter == "Healthy Only" and not info["is_healthy"]:
            continue
        if status_filter == "Diseased Only" and info["is_healthy"]:
            continue

        # Query filter
        if q:
            match_plant = q in info["plant"].lower()
            match_disease = q in info["disease"].lower()
            match_pathogen = q in info["pathogen"].lower()
            match_summary = q in info["summary"].lower()
            match_symptoms = any(q in s.lower() for s in info["symptoms"])
            if not (match_plant or match_disease or match_pathogen or match_summary or match_symptoms):
                continue

        entry = dict(info)
        entry["raw_class"] = raw_name
        results.append(entry)

    # Sort results by plant, then disease
    results.sort(key=lambda x: (x["plant"], x["disease"]))
    return results
