const fs = require("fs");
const luamin = require("luamin");

const in_path = "out.lua";
const out_path = "out.min.lua";

var data = fs.readFileSync(in_path, "utf8");
data = luamin.minify(data);
data = '-- Haxe transcendental test\n\
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
\n\
--------------------------------------------------------------------------------\n\
\n\
-- License for littleBigInt:\n\
\n\
-- MIT License\n\
--\n\
-- Copyright (c) 2020 Sylvio Sell\n\
--\n\
-- Permission is hereby granted, free of charge, to any person obtaining a copy\n\
-- of this software and associated documentation files (the "Software"), to deal\n\
-- in the Software without restriction, including without limitation the rights\n\
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n\
-- copies of the Software, and to permit persons to whom the Software is\n\
-- furnished to do so, subject to the following conditions:\n\
--\n\
-- The above copyright notice and this permission notice shall be included in all\n\
-- copies or substantial portions of the Software.\n\
--\n\
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n\
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n\
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n\
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n\
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n\
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n\
-- SOFTWARE.\n\
\n' + data + "\n";
fs.writeFileSync(out_path, data);
