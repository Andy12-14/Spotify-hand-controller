#**ENGLISH VERSION**

Spotify Hand Gesture Controller
This project provides a unique way to interact with the Spotify desktop application using hand gestures captured by your webcam. By leveraging computer vision, you can control playback, volume, and navigation without ever touching your mouse or keyboard.

##**Features**
Gesture-Based Control: Use specific hand signs to control various Spotify functions.

State-Driven Interface: Navigate between a main menu, a playback control screen, and a dedicated navigation screen.

Continuous Volume & Scrolling: Hold a gesture to continuously raise or lower the volume, or to scroll through playlists.

Visual Feedback: The webcam feed displays the current state of the application and highlights detected gestures.

Robustness: Includes a cooldown period for single-action gestures to prevent accidental spamming of commands.

Getting Started
Prerequisites
To run this application, you will need to have Python and the following libraries installed.

pip install opencv-python
pip install mediapipe
pip install numpy
pip install pyautogui

##**Usage**
Clone this repository to your local machine or download the spotify_controller.py script and the README.md file.

Open the Spotify desktop application and ensure it is the active window.

Run the Python script from your terminal:

python spotify_controller.py

Allow your operating system to access your webcam. The application window will open, and you can begin controlling Spotify with your hands. Press q to quit the application.

##**Gesture Guide**
Navigation
Enter Menu: Pinch a button on the screen with your thumb and index finger.

Return to Home: Use a Thumbs Up gesture from any screen.

Playback Controls
Play/Pause: Make a closed fist.

Raise Volume: Hold up your Index and Middle fingers.

Lower Volume: Hold up your Ring and Pinky fingers.

Seek Forward: Hold up your Middle and Ring fingers.

Seek Backward: Hold up your Index finger only.

Like Current Track: Hold up your Index and Pinky fingers.

Toggle Shuffle Mode: Hold up your Thumb, Index, and Middle fingers.

Toggle Repeat Mode: Hold up your Index, Middle, Ring, and Pinky fingers.

Scrolling
This menu is for scrolling through lists like playlists or albums.

Scroll Up: Hold up your Index finger only.

Scroll Down: Hold up your Pinky finger only.


#**VERSION  EN FRANCAIS**

Contrôleur Spotify par gestes de la main
Ce projet offre une manière unique d'interagir avec l'application de bureau Spotify en utilisant des gestes de la main capturés par votre webcam. En tirant parti de la vision par ordinateur, vous pouvez contrôler la lecture, le volume et la navigation sans jamais toucher votre souris ou votre clavier.

Fonctionnalités
Contrôle par gestes : Utilisez des signes de la main spécifiques pour contrôler diverses fonctions de Spotify.

Interface basée sur les états : Naviguez entre un menu principal, un écran de contrôle de la lecture et un écran de navigation dédié à l'aide d'un simple geste de pincement.

Volume et défilement continus : Maintenez un geste pour augmenter/diminuer le volume en continu ou pour faire défiler les listes de lecture.

Retour visuel : Le flux de la webcam affiche l'état actuel de l'application et met en évidence les gestes détectés.

Robustesse : Inclut une période de recharge pour les gestes à action unique afin d'éviter l'exécution accidentelle et rapide des commandes.

Démarrage
Prérequis
Pour exécuter cette application, vous devez avoir Python et les bibliothèques suivantes installées.

pip install opencv-python
pip install mediapipe
pip install numpy
pip install pyautogui

##**Utilisation**
Clonez ce dépôt sur votre machine locale ou téléchargez le script spotify_controller.py et le fichier README.md.

Ouvrez l'application de bureau Spotify et assurez-vous qu'elle est la fenêtre active.

Exécutez le script Python depuis votre terminal :

python spotify_controller.py

Autorisez votre système d'exploitation à accéder à votre webcam. La fenêtre de l'application s'ouvrira, et vous pourrez commencer à contrôler Spotify avec vos mains. Appuyez sur q pour quitter l'application.

Guide des gestes
Navigation
Entrer dans un menu : Pincez un bouton à l'écran avec votre pouce et votre index.

Retour à l'accueil : Faites un geste de "Pouce levé" depuis n'importe quel écran.

Contrôles de lecture
Lecture/Pause : Faites un poing fermé.

Augmenter le volume : Levez vos doigts index et majeur.

Diminuer le volume : Levez vos doigts annulaire et auriculaire.

Avance rapide : Levez vos doigts majeur et annulaire.

Retour en arrière : Levez votre doigt index uniquement.

Liker la piste : Levez vos doigts index et auriculaire.

Activer le mode aléatoire : Levez vos doigts pouce, index et majeur.

Activer le mode répéter : Levez vos doigts index, majeur, annulaire et auriculaire.

Défilement
Ce menu est pour faire défiler des listes comme les playlists ou les albums.

Défiler vers le haut : Levez votre doigt index uniquement.

Défiler vers le bas : Levez votre doigt auriculaire uniquement.
