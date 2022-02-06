# Installing dependencies:

* Install [Haxe](https://haxe.org/) 4 and Haxelib. Add them to your PATH. The specific version of Haxe used for this compilation was 4.2.4.

* Install the Haxelib package [littleBigInt](https://github.com/maitag/littleBigInt), version 0.1.3:
  ```
  haxelib install littleBigInt 0.1.3
  ```

* Install [Node.js](https://nodejs.org/).

* Install https://github.com/FATH-Mechatronics/luamin/tree/d7359250cf28ab617ba5e43d1fda6ec411b1f9f7 using npm:
  ```
  npm install FATH-Mechatronics/luamin#d7359250cf28ab617ba5e43d1fda6ec411b1f9f7
  ```

# Compilation:

* Run build.sh if you're running Linux or build.bat if you're running Windows.

* Use Node.js to run min.js:
  ```
  node min.js
  ```
