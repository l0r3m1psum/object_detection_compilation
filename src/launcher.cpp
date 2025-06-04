#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/vm.h>

#include <safetensors.hh>
#include <charconv>

const char *classes[] = {
"tench, Tinca tinca",
"goldfish, Carassius auratus",
"great white shark, white shark, man-eater, man-eating shark, Carcharodon caharias",
"tiger shark, Galeocerdo cuvieri",
"hammerhead, hammerhead shark",
"electric ray, crampfish, numbfish, torpedo",
"stingray",
"cock",
"hen",
"ostrich, Struthio camelus",
"brambling, Fringilla montifringilla",
"goldfinch, Carduelis carduelis",
"house finch, linnet, Carpodacus mexicanus",
"junco, snowbird",
"indigo bunting, indigo finch, indigo bird, Passerina cyanea",
"robin, American robin, Turdus migratorius",
"bulbul",
"jay",
"magpie",
"chickadee",
"water ouzel, dipper",
"kite",
"bald eagle, American eagle, Haliaeetus leucocephalus",
"vulture",
"great grey owl, great gray owl, Strix nebulosa",
"European fire salamander, Salamandra salamandra",
"common newt, Triturus vulgaris",
"eft",
"spotted salamander, Ambystoma maculatum",
"axolotl, mud puppy, Ambystoma mexicanum",
"bullfrog, Rana catesbeiana",
"tree frog, tree-frog",
"tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
"loggerhead, loggerhead turtle, Caretta caretta",
"leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea",
"mud turtle",
"terrapin",
"box turtle, box tortoise",
"banded gecko",
"common iguana, iguana, Iguana iguana",
"American chameleon, anole, Anolis carolinensis",
"whiptail, whiptail lizard",
"agama",
"frilled lizard, Chlamydosaurus kingi",
"alligator lizard",
"Gila monster, Heloderma suspectum",
"green lizard, Lacerta viridis",
"African chameleon, Chamaeleo chamaeleon",
"Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoeis",
"African crocodile, Nile crocodile, Crocodylus niloticus",
"American alligator, Alligator mississipiensis",
"triceratops",
"thunder snake, worm snake, Carphophis amoenus",
"ringneck snake, ring-necked snake, ring snake",
"hognose snake, puff adder, sand viper",
"green snake, grass snake",
"king snake, kingsnake",
"garter snake, grass snake",
"water snake",
"vine snake",
"night snake, Hypsiglena torquata",
"boa constrictor, Constrictor constrictor",
"rock python, rock snake, Python sebae",
"Indian cobra, Naja naja",
"green mamba",
"sea snake",
"horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
"diamondback, diamondback rattlesnake, Crotalus adamanteus",
"sidewinder, horned rattlesnake, Crotalus cerastes",
"trilobite",
"harvestman, daddy longlegs, Phalangium opilio",
"scorpion",
"black and gold garden spider, Argiope aurantia",
"barn spider, Araneus cavaticus",
"garden spider, Aranea diademata",
"black widow, Latrodectus mactans",
"tarantula",
"wolf spider, hunting spider",
"tick",
"centipede",
"black grouse",
"ptarmigan",
"ruffed grouse, partridge, Bonasa umbellus",
"prairie chicken, prairie grouse, prairie fowl",
"peacock",
"quail",
"partridge",
"African grey, African gray, Psittacus erithacus",
"macaw",
"sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
"lorikeet",
"coucal",
"bee eater",
"hornbill",
"hummingbird",
"jacamar",
"toucan",
"drake",
"red-breasted merganser, Mergus serrator",
"goose",
"black swan, Cygnus atratus",
"tusker",
"echidna, spiny anteater, anteater",
"platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhyhus anatinus",
"wallaby, brush kangaroo",
"koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus",
"wombat",
"jellyfish",
"sea anemone, anemone",
"brain coral",
"flatworm, platyhelminth",
"nematode, nematode worm, roundworm",
"conch",
"snail",
"slug",
"sea slug, nudibranch",
"chiton, coat-of-mail shell, sea cradle, polyplacophore",
"chambered nautilus, pearly nautilus, nautilus",
"Dungeness crab, Cancer magister",
"rock crab, Cancer irroratus",
"fiddler crab",
"king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodesamtschatica",
"American lobster, Northern lobster, Maine lobster, Homarus americanus",
"spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
"crayfish, crawfish, crawdad, crawdaddy",
"hermit crab",
"isopod",
"white stork, Ciconia ciconia",
"black stork, Ciconia nigra",
"spoonbill",
"flamingo",
"little blue heron, Egretta caerulea",
"American egret, great white heron, Egretta albus",
"bittern",
"crane, bird",
"limpkin, Aramus pictus",
"European gallinule, Porphyrio porphyrio",
"American coot, marsh hen, mud hen, water hen, Fulica americana",
"bustard",
"ruddy turnstone, Arenaria interpres",
"red-backed sandpiper, dunlin, Erolia alpina",
"redshank, Tringa totanus",
"dowitcher",
"oystercatcher, oyster catcher",
"pelican",
"king penguin, Aptenodytes patagonica",
"albatross, mollymawk",
"grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius rostus",
"killer whale, killer, orca, grampus, sea wolf, Orcinus orca",
"dugong, Dugong dugon",
"sea lion",
"Chihuahua",
"Japanese spaniel",
"Maltese dog, Maltese terrier, Maltese",
"Pekinese, Pekingese, Peke",
"Shih-Tzu",
"Blenheim spaniel",
"papillon",
"toy terrier",
"Rhodesian ridgeback",
"Afghan hound, Afghan",
"basset, basset hound",
"beagle",
"bloodhound, sleuthhound",
"bluetick",
"black-and-tan coonhound",
"Walker hound, Walker foxhound",
"English foxhound",
"redbone",
"borzoi, Russian wolfhound",
"Irish wolfhound",
"Italian greyhound",
"whippet",
"Ibizan hound, Ibizan Podenco",
"Norwegian elkhound, elkhound",
"otterhound, otter hound",
"Saluki, gazelle hound",
"Scottish deerhound, deerhound",
"Weimaraner",
"Staffordshire bullterrier, Staffordshire bull terrier",
"American Staffordshire terrier, Staffordshire terrier, American pit bull rrier, pit bull terrier",
"Bedlington terrier",
"Border terrier",
"Kerry blue terrier",
"Irish terrier",
"Norfolk terrier",
"Norwich terrier",
"Yorkshire terrier",
"wire-haired fox terrier",
"Lakeland terrier",
"Sealyham terrier, Sealyham",
"Airedale, Airedale terrier",
"cairn, cairn terrier",
"Australian terrier",
"Dandie Dinmont, Dandie Dinmont terrier",
"Boston bull, Boston terrier",
"miniature schnauzer",
"giant schnauzer",
"standard schnauzer",
"Scotch terrier, Scottish terrier, Scottie",
"Tibetan terrier, chrysanthemum dog",
"silky terrier, Sydney silky",
"soft-coated wheaten terrier",
"West Highland white terrier",
"Lhasa, Lhasa apso",
"flat-coated retriever",
"curly-coated retriever",
"golden retriever",
"Labrador retriever",
"Chesapeake Bay retriever",
"German short-haired pointer",
"vizsla, Hungarian pointer",
"English setter",
"Irish setter, red setter",
"Gordon setter",
"Brittany spaniel",
"clumber, clumber spaniel",
"English springer, English springer spaniel",
"Welsh springer spaniel",
"cocker spaniel, English cocker spaniel, cocker",
"Sussex spaniel",
"Irish water spaniel",
"kuvasz",
"schipperke",
"groenendael",
"malinois",
"briard",
"kelpie",
"komondor",
"Old English sheepdog, bobtail",
"Shetland sheepdog, Shetland sheep dog, Shetland",
"collie",
"Border collie",
"Bouvier des Flandres, Bouviers des Flandres",
"Rottweiler",
"German shepherd, German shepherd dog, German police dog, alsatian",
"Doberman, Doberman pinscher",
"miniature pinscher",
"Greater Swiss Mountain dog",
"Bernese mountain dog",
"Appenzeller",
"EntleBucher",
"boxer",
"bull mastiff",
"Tibetan mastiff",
"French bulldog",
"Great Dane",
"Saint Bernard, St Bernard",
"Eskimo dog, husky",
"malamute, malemute, Alaskan malamute",
"Siberian husky",
"dalmatian, coach dog, carriage dog",
"affenpinscher, monkey pinscher, monkey dog",
"basenji",
"pug, pug-dog",
"Leonberg",
"Newfoundland, Newfoundland dog",
"Great Pyrenees",
"Samoyed, Samoyede",
"Pomeranian",
"chow, chow chow",
"keeshond",
"Brabancon griffon",
"Pembroke, Pembroke Welsh corgi",
"Cardigan, Cardigan Welsh corgi",
"toy poodle",
"miniature poodle",
"standard poodle",
"Mexican hairless",
"timber wolf, grey wolf, gray wolf, Canis lupus",
"white wolf, Arctic wolf, Canis lupus tundrarum",
"red wolf, maned wolf, Canis rufus, Canis niger",
"coyote, prairie wolf, brush wolf, Canis latrans",
"dingo, warrigal, warragal, Canis dingo",
"dhole, Cuon alpinus",
"African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus",
"hyena, hyaena",
"red fox, Vulpes vulpes",
"kit fox, Vulpes macrotis",
"Arctic fox, white fox, Alopex lagopus",
"grey fox, gray fox, Urocyon cinereoargenteus",
"tabby, tabby cat",
"tiger cat",
"Persian cat",
"Siamese cat, Siamese",
"Egyptian cat",
"cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
"lynx, catamount",
"leopard, Panthera pardus",
"snow leopard, ounce, Panthera uncia",
"jaguar, panther, Panthera onca, Felis onca",
"lion, king of beasts, Panthera leo",
"tiger, Panthera tigris",
"cheetah, chetah, Acinonyx jubatus",
"brown bear, bruin, Ursus arctos",
"American black bear, black bear, Ursus americanus, Euarctos americanus",
"ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus",
"sloth bear, Melursus ursinus, Ursus ursinus",
"mongoose",
"meerkat, mierkat",
"tiger beetle",
"ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
"ground beetle, carabid beetle",
"long-horned beetle, longicorn, longicorn beetle",
"leaf beetle, chrysomelid",
"dung beetle",
"rhinoceros beetle",
"weevil",
"fly",
"bee",
"ant, emmet, pismire",
"grasshopper, hopper",
"cricket",
"walking stick, walkingstick, stick insect",
"cockroach, roach",
"mantis, mantid",
"cicada, cicala",
"leafhopper",
"lacewing, lacewing fly",
"dragonfly, darning needle, devil's darning needle, sewing needle, snake fder, snake doctor, mosquito hawk, skeeter hawk,",
"damselfly",
"admiral",
"ringlet, ringlet butterfly",
"monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
"cabbage butterfly",
"sulphur butterfly, sulfur butterfly",
"lycaenid, lycaenid butterfly",
"starfish, sea star",
"sea urchin",
"sea cucumber, holothurian",
"wood rabbit, cottontail, cottontail rabbit",
"hare",
"Angora, Angora rabbit",
"hamster",
"porcupine, hedgehog",
"fox squirrel, eastern fox squirrel, Sciurus niger",
"marmot",
"beaver",
"guinea pig, Cavia cobaya",
"sorrel",
"zebra",
"hog, pig, grunter, squealer, Sus scrofa",
"wild boar, boar, Sus scrofa",
"warthog",
"hippopotamus, hippo, river horse, Hippopotamus amphibius",
"ox",
"water buffalo, water ox, Asiatic buffalo, Bubalus bubalis",
"bison",
"ram, tup",
"bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain eep, Ovis canadensis",
"ibex, Capra ibex",
"hartebeest",
"impala, Aepyceros melampus",
"gazelle",
"Arabian camel, dromedary, Camelus dromedarius",
"llama",
"weasel",
"mink",
"polecat, fitch, foulmart, foumart, Mustela putorius",
"black-footed ferret, ferret, Mustela nigripes",
"otter",
"skunk, polecat, wood pussy",
"badger",
"armadillo",
"three-toed sloth, ai, Bradypus tridactylus",
"orangutan, orang, orangutang, Pongo pygmaeus",
"gorilla, Gorilla gorilla",
"chimpanzee, chimp, Pan troglodytes",
"gibbon, Hylobates lar",
"siamang, Hylobates syndactylus, Symphalangus syndactylus",
"guenon, guenon monkey",
"patas, hussar monkey, Erythrocebus patas",
"baboon",
"macaque",
"langur",
"colobus, colobus monkey",
"proboscis monkey, Nasalis larvatus",
"marmoset",
"capuchin, ringtail, Cebus capucinus",
"howler monkey, howler",
"titi, titi monkey",
"spider monkey, Ateles geoffroyi",
"squirrel monkey, Saimiri sciureus",
"Madagascar cat, ring-tailed lemur, Lemur catta",
"indri, indris, Indri indri, Indri brevicaudatus",
"Indian elephant, Elephas maximus",
"African elephant, Loxodonta africana",
"lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
"giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca",
"barracouta, snoek",
"eel",
"coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch",
"rock beauty, Holocanthus tricolor",
"anemone fish",
"sturgeon",
"gar, garfish, garpike, billfish, Lepisosteus osseus",
"lionfish",
"puffer, pufferfish, blowfish, globefish",
"abacus",
"abaya",
"academic gown, academic robe, judge's robe",
"accordion, piano accordion, squeeze box",
"acoustic guitar",
"aircraft carrier, carrier, flattop, attack aircraft carrier",
"airliner",
"airship, dirigible",
"altar",
"ambulance",
"amphibian, amphibious vehicle",
"analog clock",
"apiary, bee house",
"apron",
"ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustb, trash barrel, trash bin",
"assault rifle, assault gun",
"backpack, back pack, knapsack, packsack, rucksack, haversack",
"bakery, bakeshop, bakehouse",
"balance beam, beam",
"balloon",
"ballpoint, ballpoint pen, ballpen, Biro",
"Band Aid",
"banjo",
"bannister, banister, balustrade, balusters, handrail",
"barbell",
"barber chair",
"barbershop",
"barn",
"barometer",
"barrel, cask",
"barrow, garden cart, lawn cart, wheelbarrow",
"baseball",
"basketball",
"bassinet",
"bassoon",
"bathing cap, swimming cap",
"bath towel",
"bathtub, bathing tub, bath, tub",
"beach wagon, station wagon, wagon, estate car, beach waggon, station wagg, waggon",
"beacon, lighthouse, beacon light, pharos",
"beaker",
"bearskin, busby, shako",
"beer bottle",
"beer glass",
"bell cote, bell cot",
"bib",
"bicycle-built-for-two, tandem bicycle, tandem",
"bikini, two-piece",
"binder, ring-binder",
"binoculars, field glasses, opera glasses",
"birdhouse",
"boathouse",
"bobsled, bobsleigh, bob",
"bolo tie, bolo, bola tie, bola",
"bonnet, poke bonnet",
"bookcase",
"bookshop, bookstore, bookstall",
"bottlecap",
"bow",
"bow tie, bow-tie, bowtie",
"brass, memorial tablet, plaque",
"brassiere, bra, bandeau",
"breakwater, groin, groyne, mole, bulwark, seawall, jetty",
"breastplate, aegis, egis",
"broom",
"bucket, pail",
"buckle",
"bulletproof vest",
"bullet train, bullet",
"butcher shop, meat market",
"cab, hack, taxi, taxicab",
"caldron, cauldron",
"candle, taper, wax light",
"cannon",
"canoe",
"can opener, tin opener",
"cardigan",
"car mirror",
"carousel, carrousel, merry-go-round, roundabout, whirligig",
"carpenter's kit, tool kit",
"carton",
"car wheel",
"cash machine, cash dispenser, automated teller machine, automatic teller chine, automated teller, automatic teller, ATM",
"cassette",
"cassette player",
"castle",
"catamaran",
"CD player",
"cello, violoncello",
"cellular telephone, cellular phone, cellphone, cell, mobile phone",
"chain",
"chainlink fence",
"chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring mour",
"chain saw, chainsaw",
"chest",
"chiffonier, commode",
"chime, bell, gong",
"china cabinet, china closet",
"Christmas stocking",
"church, church building",
"cinema, movie theater, movie theatre, movie house, picture palace",
"cleaver, meat cleaver, chopper",
"cliff dwelling",
"cloak",
"clog, geta, patten, sabot",
"cocktail shaker",
"coffee mug",
"coffeepot",
"coil, spiral, volute, whorl, helix",
"combination lock",
"computer keyboard, keypad",
"confectionery, confectionary, candy store",
"container ship, containership, container vessel",
"convertible",
"corkscrew, bottle screw",
"cornet, horn, trumpet, trump",
"cowboy boot",
"cowboy hat, ten-gallon hat",
"cradle",
"crane",
"crash helmet",
"crate",
"crib, cot",
"Crock Pot",
"croquet ball",
"crutch",
"cuirass",
"dam, dike, dyke",
"desk",
"desktop computer",
"dial telephone, dial phone",
"diaper, nappy, napkin",
"digital clock",
"digital watch",
"dining table, board",
"dishrag, dishcloth",
"dishwasher, dish washer, dishwashing machine",
"disk brake, disc brake",
"dock, dockage, docking facility",
"dogsled, dog sled, dog sleigh",
"dome",
"doormat, welcome mat",
"drilling platform, offshore rig",
"drum, membranophone, tympan",
"drumstick",
"dumbbell",
"Dutch oven",
"electric fan, blower",
"electric guitar",
"electric locomotive",
"entertainment center",
"envelope",
"espresso maker",
"face powder",
"feather boa, boa",
"file, file cabinet, filing cabinet",
"fireboat",
"fire engine, fire truck",
"fire screen, fireguard",
"flagpole, flagstaff",
"flute, transverse flute",
"folding chair",
"football helmet",
"forklift",
"fountain",
"fountain pen",
"four-poster",
"freight car",
"French horn, horn",
"frying pan, frypan, skillet",
"fur coat",
"garbage truck, dustcart",
"gasmask, respirator, gas helmet",
"gas pump, gasoline pump, petrol pump, island dispenser",
"goblet",
"go-kart",
"golf ball",
"golfcart, golf cart",
"gondola",
"gong, tam-tam",
"gown",
"grand piano, grand",
"greenhouse, nursery, glasshouse",
"grille, radiator grille",
"grocery store, grocery, food market, market",
"guillotine",
"hair slide",
"hair spray",
"half track",
"hammer",
"hamper",
"hand blower, blow dryer, blow drier, hair dryer, hair drier",
"hand-held computer, hand-held microcomputer",
"handkerchief, hankie, hanky, hankey",
"hard disc, hard disk, fixed disk",
"harmonica, mouth organ, harp, mouth harp",
"harp",
"harvester, reaper",
"hatchet",
"holster",
"home theater, home theatre",
"honeycomb",
"hook, claw",
"hoopskirt, crinoline",
"horizontal bar, high bar",
"horse cart, horse-cart",
"hourglass",
"iPod",
"iron, smoothing iron",
"jack-o'-lantern",
"jean, blue jean, denim",
"jeep, landrover",
"jersey, T-shirt, tee shirt",
"jigsaw puzzle",
"jinrikisha, ricksha, rickshaw",
"joystick",
"kimono",
"knee pad",
"knot",
"lab coat, laboratory coat",
"ladle",
"lampshade, lamp shade",
"laptop, laptop computer",
"lawn mower, mower",
"lens cap, lens cover",
"letter opener, paper knife, paperknife",
"library",
"lifeboat",
"lighter, light, igniter, ignitor",
"limousine, limo",
"liner, ocean liner",
"lipstick, lip rouge",
"Loafer",
"lotion",
"loudspeaker, speaker, speaker unit, loudspeaker system, speaker system",
"loupe, jeweler's loupe",
"lumbermill, sawmill",
"magnetic compass",
"mailbag, postbag",
"mailbox, letter box",
"maillot",
"maillot, tank suit",
"manhole cover",
"maraca",
"marimba, xylophone",
"mask",
"matchstick",
"maypole",
"maze, labyrinth",
"measuring cup",
"medicine chest, medicine cabinet",
"megalith, megalithic structure",
"microphone, mike",
"microwave, microwave oven",
"military uniform",
"milk can",
"minibus",
"miniskirt, mini",
"minivan",
"missile",
"mitten",
"mixing bowl",
"mobile home, manufactured home",
"Model T",
"modem",
"monastery",
"monitor",
"moped",
"mortar",
"mortarboard",
"mosque",
"mosquito net",
"motor scooter, scooter",
"mountain bike, all-terrain bike, off-roader",
"mountain tent",
"mouse, computer mouse",
"mousetrap",
"moving van",
"muzzle",
"nail",
"neck brace",
"necklace",
"nipple",
"notebook, notebook computer",
"obelisk",
"oboe, hautboy, hautbois",
"ocarina, sweet potato",
"odometer, hodometer, mileometer, milometer",
"oil filter",
"organ, pipe organ",
"oscilloscope, scope, cathode-ray oscilloscope, CRO",
"overskirt",
"oxcart",
"oxygen mask",
"packet",
"paddle, boat paddle",
"paddlewheel, paddle wheel",
"padlock",
"paintbrush",
"pajama, pyjama, pj's, jammies",
"palace",
"panpipe, pandean pipe, syrinx",
"paper towel",
"parachute, chute",
"parallel bars, bars",
"park bench",
"parking meter",
"passenger car, coach, carriage",
"patio, terrace",
"pay-phone, pay-station",
"pedestal, plinth, footstall",
"pencil box, pencil case",
"pencil sharpener",
"perfume, essence",
"Petri dish",
"photocopier",
"pick, plectrum, plectron",
"pickelhaube",
"picket fence, paling",
"pickup, pickup truck",
"pier",
"piggy bank, penny bank",
"pill bottle",
"pillow",
"ping-pong ball",
"pinwheel",
"pirate, pirate ship",
"pitcher, ewer",
"plane, carpenter's plane, woodworking plane",
"planetarium",
"plastic bag",
"plate rack",
"plow, plough",
"plunger, plumber's helper",
"Polaroid camera, Polaroid Land camera",
"pole",
"police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria",
"poncho",
"pool table, billiard table, snooker table",
"pop bottle, soda bottle",
"pot, flowerpot",
"potter's wheel",
"power drill",
"prayer rug, prayer mat",
"printer",
"prison, prison house",
"projectile, missile",
"projector",
"puck, hockey puck",
"punching bag, punch bag, punching ball, punchball",
"purse",
"quill, quill pen",
"quilt, comforter, comfort, puff",
"racer, race car, racing car",
"racket, racquet",
"radiator",
"radio, wireless",
"radio telescope, radio reflector",
"rain barrel",
"recreational vehicle, RV, R.V.",
"reel",
"reflex camera",
"refrigerator, icebox",
"remote control, remote",
"restaurant, eating house, eating place, eatery",
"revolver, six-gun, six-shooter",
"rifle",
"rocking chair, rocker",
"rotisserie",
"rubber eraser, rubber, pencil eraser",
"rugby ball",
"rule, ruler",
"running shoe",
"safe",
"safety pin",
"saltshaker, salt shaker",
"sandal",
"sarong",
"sax, saxophone",
"scabbard",
"scale, weighing machine",
"school bus",
"schooner",
"scoreboard",
"screen, CRT screen",
"screw",
"screwdriver",
"seat belt, seatbelt",
"sewing machine",
"shield, buckler",
"shoe shop, shoe-shop, shoe store",
"shoji",
"shopping basket",
"shopping cart",
"shovel",
"shower cap",
"shower curtain",
"ski",
"ski mask",
"sleeping bag",
"slide rule, slipstick",
"sliding door",
"slot, one-armed bandit",
"snorkel",
"snowmobile",
"snowplow, snowplough",
"soap dispenser",
"soccer ball",
"sock",
"solar dish, solar collector, solar furnace",
"sombrero",
"soup bowl",
"space bar",
"space heater",
"space shuttle",
"spatula",
"speedboat",
"spider web, spider's web",
"spindle",
"sports car, sport car",
"spotlight, spot",
"stage",
"steam locomotive",
"steel arch bridge",
"steel drum",
"stethoscope",
"stole",
"stone wall",
"stopwatch, stop watch",
"stove",
"strainer",
"streetcar, tram, tramcar, trolley, trolley car",
"stretcher",
"studio couch, day bed",
"stupa, tope",
"submarine, pigboat, sub, U-boat",
"suit, suit of clothes",
"sundial",
"sunglass",
"sunglasses, dark glasses, shades",
"sunscreen, sunblock, sun blocker",
"suspension bridge",
"swab, swob, mop",
"sweatshirt",
"swimming trunks, bathing trunks",
"swing",
"switch, electric switch, electrical switch",
"syringe",
"table lamp",
"tank, army tank, armored combat vehicle, armoured combat vehicle",
"tape player",
"teapot",
"teddy, teddy bear",
"television, television system",
"tennis ball",
"thatch, thatched roof",
"theater curtain, theatre curtain",
"thimble",
"thresher, thrasher, threshing machine",
"throne",
"tile roof",
"toaster",
"tobacco shop, tobacconist shop, tobacconist",
"toilet seat",
"torch",
"totem pole",
"tow truck, tow car, wrecker",
"toyshop",
"tractor",
"trailer truck, tractor trailer, trucking rig, rig, articulated lorry, sem,",
"tray",
"trench coat",
"tricycle, trike, velocipede",
"trimaran",
"tripod",
"triumphal arch",
"trolleybus, trolley coach, trackless trolley",
"trombone",
"tub, vat",
"turnstile",
"typewriter keyboard",
"umbrella",
"unicycle, monocycle",
"upright, upright piano",
"vacuum, vacuum cleaner",
"vase",
"vault",
"velvet",
"vending machine",
"vestment",
"viaduct",
"violin, fiddle",
"volleyball",
"waffle iron",
"wall clock",
"wallet, billfold, notecase, pocketbook",
"wardrobe, closet, press",
"warplane, military plane",
"washbasin, handbasin, washbowl, lavabo, wash-hand basin",
"washer, automatic washer, washing machine",
"water bottle",
"water jug",
"water tower",
"whiskey jug",
"whistle",
"wig",
"window screen",
"window shade",
"Windsor tie",
"wine bottle",
"wing",
"wok",
"wooden spoon",
"wool, woolen, woollen",
"worm fence, snake fence, snake-rail fence, Virginia fence",
"wreck",
"yawl",
"yurt",
"web site, website, internet site, site",
"comic book",
"crossword puzzle, crossword",
"street sign",
"traffic light, traffic signal, stoplight",
"book jacket, dust cover, dust jacket, dust wrapper",
"menu",
"plate",
"guacamole",
"consomme",
"hot pot, hotpot",
"trifle",
"ice cream, icecream",
"ice lolly, lolly, lollipop, popsicle",
"French loaf",
"bagel, beigel",
"pretzel",
"cheeseburger",
"hotdog, hot dog, red hot",
"mashed potato",
"head cabbage",
"broccoli",
"cauliflower",
"zucchini, courgette",
"spaghetti squash",
"acorn squash",
"butternut squash",
"cucumber, cuke",
"artichoke, globe artichoke",
"bell pepper",
"cardoon",
"mushroom",
"Granny Smith",
"strawberry",
"orange",
"lemon",
"fig",
"pineapple, ananas",
"banana",
"jackfruit, jak, jack",
"custard apple",
"pomegranate",
"hay",
"carbonara",
"chocolate sauce, chocolate syrup",
"dough",
"meat loaf, meatloaf",
"pizza, pizza pie",
"potpie",
"burrito",
"red wine",
"espresso",
"cup",
"eggnog",
"alp",
"bubble",
"cliff, drop, drop-off",
"coral reef",
"geyser",
"lakeside, lakeshore",
"promontory, headland, head, foreland",
"sandbar, sand bar",
"seashore, coast, seacoast, sea-coast",
"valley, vale",
"volcano",
"ballplayer, baseball player",
"groom, bridegroom",
"scuba diver",
"rapeseed",
"daisy",
"yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripium parviflorum,",
"corn",
"acorn",
"hip, rose hip, rosehip",
"buckeye, horse chestnut, conker",
"coral fungus",
"agaric",
"gyromitra",
"stinkhorn, carrion fungus",
"earthstar",
"hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa",
"bolete",
"ear, spike, capitulum",
"toilet tissue, toilet paper, bathroom tissue",
};

static std::ostream&
operator<<(std::ostream& os, const std::vector<int64_t>& vec) {
  os << '{';
  for (size_t i = 0, e = vec.size(); i != e; ++i) {
    if (i != 0) os << ", ";
    os << vec[i];
  }
  os << '}';
  return os;
}

// tvm::runtime::LoadVMModule seems to be brocken (for some MSVC bug?) so here
// there is a reimplementation.
static tvm::runtime::Module
LoadVMModule(const std::string& path, tvm::Device device) {
  using namespace tvm::runtime;
  Module dso_mod = Module::LoadFromFile(path, "");
  PackedFunc vm_load_executable = dso_mod.GetFunction("vm_load_executable");
  CHECK(vm_load_executable != nullptr)
      << "ValueError: File `" << path
      << "` is not built by RelaxVM, because `vm_load_executable` does not exist";
  Module mod = vm_load_executable();
  PackedFunc vm_initialization = mod.GetFunction("vm_initialization");
  CHECK(vm_initialization != nullptr)
      << "ValueError: File `" << path
      << "` is not built by RelaxVM, because `vm_initialization` does not exist";
  vm_initialization(
    static_cast<int>(device.device_type), static_cast<int>(device.device_id), static_cast<int>(AllocatorType::kPooled),
    static_cast<int>(kDLCPU),             0,                                  static_cast<int>(AllocatorType::kPooled)
  );
  return mod;
}

static void
manually_load_module() {
  // runtime.module.loadbinary_cuda
  const char *loader_name = "runtime.module.loadfile_so";
  const tvm::runtime::PackedFunc *pf = tvm::runtime::Registry::Get(loader_name);
  if (!pf) {
    std::cerr << "cannot find " << loader_name << '\n';
    abort();
  }
  const char *file_so = "build/mlp.dll";
  tvm::runtime::TVMRetValue rv = (*pf)(file_so);
  if (rv.type_code() != kTVMModuleHandle) {
    std::cerr << "cannot load " << file_so << '\n';
    abort();
  }
  tvm::runtime::Module mod = rv.operator tvm::runtime::Module();

  std::cout << mod->type_key() << '\n';

  tvm::runtime::ModuleNode *mod_node = mod.operator->();

  tvm::runtime::relax_vm::VMExecutable *vm = static_cast<tvm::runtime::relax_vm::VMExecutable *>(mod_node);
  // std::cout << vm->AsText() << '\n';

  std::vector<tvm::runtime::relax_vm::VMFuncInfo> func_table = vm->func_table;

  std::cout << func_table.size() << '\n';
  for (const auto& func_info : func_table) {
    std::cout << func_info.name << '\n';
  }

  // Il modulo esportato espone solo le "sigole" funzioni, non la forward, per
  // ottenere quelle funzioni il modulo va inizializzato con vm_load_executable
  // e vm_initialization.
  // >dumpbin /exports build\mlp.dll
  const char *function_name = "add";
  tvm::runtime::PackedFunc func = (*mod_node).GetFunction(function_name, true);
  if (!func.defined()) {
    std::cerr << "cannot get " << function_name << '\n';
  }
  else {
    static float my_a[256], my_b[256], my_c[256];

    for (int i = 0; i < 256; i++) {
        my_a[i] = 1.0f;
        my_b[i] = 2.0f;
    }
    tvm::runtime::NDArray a = tvm::runtime::NDArray::Empty(
        { 1, 256 }, // outpur of matmul has two dimensions
        DLDataType{ kDLFloat, 32, 1 },
        DLDevice{ kDLCPU, 0 }
    );
    tvm::runtime::NDArray b = tvm::runtime::NDArray::Empty(
        { 256 },
        DLDataType{ kDLFloat, 32, 1 },
        DLDevice{ kDLCPU, 0 }
    );
    tvm::runtime::NDArray c = tvm::runtime::NDArray::Empty(
        { 1, 256 },
        DLDataType{ kDLFloat, 32, 1 },
        DLDevice{ kDLCPU, 0 }
    );

    a.CopyFromBytes(my_a, sizeof my_a);
    b.CopyFromBytes(my_b, sizeof my_b);
    func(a, b, c); // DPS
    c.CopyToBytes(my_c, sizeof my_c);

    std::cout << my_c[0] << '\n';
  }
}

static bool
ends_with(std::string const& value, std::string const& ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.crbegin(), ending.crend(), value.crbegin());
}

static size_t
argmax(const float *data, size_t size) {
  size_t res = 0;
  for (size_t i = 0; i < size; ++i) {
    if (data[i] > data[res]) {
      res = i;
    }
  }
  return res;
}

static DLDataType
dtype_to_DLDataType(safetensors::dtype dtype) {
  switch (dtype) {
  case safetensors::dtype::kBOOL:     return DLDataType{ kDLBool, 8, 1 };
  case safetensors::dtype::kUINT8:    return DLDataType{ kDLUInt, 8, 1 };
  case safetensors::dtype::kINT8:     return DLDataType{ kDLInt ,8, 1 };
  case safetensors::dtype::kINT16:    return DLDataType{ kDLInt, 16, 1 };
  case safetensors::dtype::kUINT16:   return DLDataType{ kDLUInt, 16, 1 };
  case safetensors::dtype::kFLOAT16:  return DLDataType{ kDLFloat, 16, 1 };
  case safetensors::dtype::kBFLOAT16: return DLDataType{ kDLBfloat, 16, 1 };
  case safetensors::dtype::kINT32:    return DLDataType{ kDLInt, 32, 1 };
  case safetensors::dtype::kUINT32:   return DLDataType{ kDLUInt, 32, 1 };
  case safetensors::dtype::kFLOAT32:  return DLDataType{ kDLFloat, 32, 1 };
  case safetensors::dtype::kFLOAT64:  return DLDataType{ kDLFloat, 64, 1 };
  case safetensors::dtype::kINT64:    return DLDataType{ kDLInt, 64, 1 };
  case safetensors::dtype::kUINT64:   return DLDataType{ kDLUInt, 64, 1 };
  }
}

// #include <intrin.h>

// #pragma intrinsic(__rdtsc)

int main() {
  std::string file_so = "build\\resnet18.dll";
  tvm::runtime::Module mod = LoadVMModule(file_so, tvm::Device{kDLCUDA, 0});
  std::cout << mod->type_key() << '\n';
  tvm::runtime::PackedFunc forward = mod.GetFunction("main", false);
  CHECK(forward != nullptr) << "cannot get forward";

  {
    tvm::runtime::ModuleNode *mod_node = mod.operator->();
    tvm::runtime::relax_vm::VirtualMachine *vm = static_cast<tvm::runtime::relax_vm::VirtualMachine *>(mod_node);
    tvm::runtime::relax_vm::VMExecutable *ex = static_cast<tvm::runtime::relax_vm::VMExecutable *>(mod_node);
    // __debugbreak();
    //tvm::String test = (*ex).GetFunction("as_text", true)();
#if 0
    tvm::runtime::PackedFunc as_text = ex->GetFunction(
      "as_text",
      static_cast<tvm::runtime::Object>(*ex)
    );
    tvm::String test = as_text();
#endif
  }

  std::string file_safetensors = "build\\resnet18.safetensors";
  safetensors::safetensors_t weights{};
  {
    std::string warn{}, err{};
    bool ok = safetensors::mmap_from_file(file_safetensors, &weights, &warn, &err);
    LOG_IF(WARNING, warn != "") << warn;
    CHECK(ok) << "cannot load weights: " << err;
  }

  int num_args = weights.tensors.size() + 1;
  TVMValue *values = new TVMValue[num_args];
  int *type_codes = new int[num_args];
  tvm::runtime::TVMArgsSetter setter(values, type_codes);
  tvm::runtime::NDArray *input_and_params = new tvm::runtime::NDArray[num_args]; // I hate C++

  {
    bool found = false;
    std::string position;
    size_t position_index = 0;
    std::from_chars_result res;
    safetensors::tensor_t tmp_tensor{};
    DLDevice device{ kDLCUDA, 0 };
    DLDataType datatype{ kDLFloat, 32, 1 };

    std::vector<tvm::runtime::ShapeTuple::index_type> shape_vec{1, 3, 224, 224};
    input_and_params[0] = tvm::runtime::NDArray::Empty(shape_vec, datatype, device);
    setter(0, input_and_params[0]);

    for (int i = 1; i < num_args; i++) {
      found = weights.tensors.at(i-1, &tmp_tensor);
      CHECK(found);
      CHECK(tmp_tensor.dtype == safetensors::dtype::kFLOAT32);

      const std::string& key = weights.tensors.keys()[i-1];
      found = weights.metadata.at(key, &position);
      CHECK(found) << "cannot determine where to put " << key << " from the metadata";
      res = std::from_chars(position.data(), position.data()+position.size(), position_index);
      CHECK(res.ec == std::errc()) << "position is not a number";

      shape_vec.resize(tmp_tensor.shape.size());
      for (size_t i = 0; i < shape_vec.size(); i++) shape_vec[i] = tmp_tensor.shape[i];
      // std::cout << key << ' ' << shape_vec << '\n';

      input_and_params[position_index+1] = tvm::runtime::NDArray::Empty(shape_vec, datatype, device);
      input_and_params[position_index+1].CopyFromBytes(
        weights.databuffer_addr + tmp_tensor.data_offsets[0],
        tmp_tensor.data_offsets[1] - tmp_tensor.data_offsets[0]
      );
      setter(position_index+1, input_and_params[position_index+1]);
    }
  }

  tvm::runtime::TVMArgs args(values, type_codes, num_args);
  tvm::runtime::TVMRetValue rv;
  {
    std::cout << "Making 1000 inferences\n";
    clock_t start = clock();
    for (size_t i = 0; i < 1000; ++i) {
      forward.CallPacked(args, &rv);
    }
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    std::cout << "seconds: " << seconds << '\n';
  }
  CHECK(rv.type_code() == kTVMObjectHandle);

  tvm::runtime::ObjectRef res = rv.operator tvm::runtime::ObjectRef();
  CHECK(res.defined());

  tvm::runtime::Optional<tvm::runtime::Array<tvm::runtime::NDArray>> opt_array;
  opt_array = res.as<tvm::runtime::Array<tvm::runtime::NDArray>>();
  CHECK(opt_array);
  tvm::runtime::Array<tvm::runtime::NDArray> array = opt_array.value();
  for (size_t i = 0; i < array.size(); ++i) {
    tvm::runtime::NDArray elem_gpu = array[0];
    tvm::runtime::NDArray elem = tvm::runtime::NDArray::Empty(elem_gpu.Shape(), { kDLFloat, 32, 1 }, { kDLCPU, 0 });
    elem_gpu.CopyTo(elem);
    std::cout << elem.DataType() << ' ' << elem.Shape() << '\n';
    const float *data = (const float *) elem->data;
    size_t j = argmax(data, 1000);
    std::cout << classes[j] << '\n';
  }

  if (false) {
    DLDevice device{ kDLCUDA, 0 };
    DLDataType datatype{ kDLFloat, 32, 1 };
    tvm::runtime::NDArray x  = tvm::runtime::NDArray::Empty({1, 784}, datatype, device);
    tvm::runtime::NDArray w1 = tvm::runtime::NDArray::Empty({256, 784}, datatype, device);
    tvm::runtime::NDArray b1 = tvm::runtime::NDArray::Empty({256}, datatype, device);
    tvm::runtime::NDArray w2 = tvm::runtime::NDArray::Empty({10, 256}, datatype, device);
    tvm::runtime::NDArray y  = tvm::runtime::NDArray::Empty({1, 10}, datatype, device);

    tvm::runtime::TVMRetValue rv = forward(x, w1, b1, w2);
    if (rv.type_code() != kTVMNDArrayHandle) {
      std::cerr << "not an array\n";
      return 1;
    }
    tvm::runtime::NDArray res = rv.operator tvm::runtime::NDArray();
  }

  return 0;
}
