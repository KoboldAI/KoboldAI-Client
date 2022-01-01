const fs = require("fs");
const luamin = require("luamin");

const in_path = "out.lua";
const out_path = "out.min.lua";

var data = fs.readFileSync(in_path, "utf8");
data = luamin.minify(data);
data = "-- Haxe trancendental test\n\
-- This is a script written in Haxe that prints the natural logarithm of the\n\
-- golden ratio in base 10 to arbitrarily many digits.\n\
\n\
-- This file is part of KoboldAI.\n\
--\n\
-- KoboldAI is free software: you can redistribute it and/or modify\n\
-- it under the terms of the GNU Affero General Public License as published by\n\
-- the Free Software Foundation, either version 3 of the License, or\n\
-- (at your option) any later version.\n\
--\n\
-- This program is distributed in the hope that it will be useful,\n\
-- but WITHOUT ANY WARRANTY; without even the implied warranty of\n\
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n\
-- GNU Affero General Public License for more details.\n\
--\n\
-- You should have received a copy of the GNU Affero General Public License\n\
-- along with this program.  If not, see <https://www.gnu.org/licenses/>.\n\
\n" + data + "\n";
fs.writeFileSync(out_path, data);
