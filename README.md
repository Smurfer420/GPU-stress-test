# GPU Stress tester
Wanna find out how good your GPU is? then this will help you with it

## Features

- Detail changing (eg: "`--detail 4`") (1-9 recommended)
- Instance amount changing (eg: "`--instances 250`") (depending on the detail, 1-10000 recommended
- FPS limiter (eg "`--fps 60`". (if you dont set it, it will be set on unlimited)
- Heatmap (eg "`--heatmap`")
- FXAA (eg "`--fxaa`")

## Usage
Linux (and most likely MAC os too): 
- 1: Download the `main.c`
- 2: run `sudo apt install gcc` in your terminal
- 3: go into the directory with the `main.c` and run `gcc main.c -o main -lGL -lglfw -lGLEW -lm -lcglm`
- 4: run ./main {parameters} (eg: "`./main --detail 4 --instances 9 --fps 120 --fxaa`")

or download the `main` file and skip steps 1-3

Windows (i believe / hope):
Option 1:
- 1: Download the `main.exe`
- 2: double click the `main.exe`

Option 2 (recommended):
- 1: download the `main.exe`
- 2: right click in the folder / desktop the file is in, and click the option to open in command prompt
- 3: type `main.exe {parameters}`

## FAQ

#### Windows version?

~uhh, no? (compile it to an .exe yourself, i am to lazy)~ Now working. (should have the same performance as the linux version too)

#### Macos version?

idk. i dont have a mac to test it. and i think the linux version works fine, cause linux and mac are both children of the Unix family

#### Can i modify, share, or use the code?

Yes, you may use, modify or share this code without asking me.

#### How can i modify this program for my own uses?

Download the source code from this repo (`main.c`), and open this file in your desired code editor ([VSCode](https://code.visualstudio.com/download) recommended)

## Authors

- Just me, Smurfer420

### Contacts

- Discord: Smurfer420
- Email: i would provide one, if i would check them on a regular basis
