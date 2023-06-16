import sys

header = r'''
\documentclass[border=5pt,tikz]{standalone}
\usepackage{tikz-3dplot}
\usetikzlibrary{3d}
\begin{document}
\tdplotsetmaincoords{60}{135}
\begin{tikzpicture}
    \clip (-8,-6) rectangle (8,6);
    \begin{scope}[tdplot_main_coords]

  % Define grid dimensions
  \def\gridRows{6}
  \def\gridCols{6}
  % Define rectangle dimensions
  \def\rectWidth{1}
  \def\rectHeight{1}
'''

footer = r'''
  % Draw vertical lines
  \foreach \col in {0,...,\gridCols} {
    \draw[canvas is zy plane at x=6,gray,very thin] (\col*\rectWidth, 0) -- ++(0, \gridRows*\rectHeight);
    \draw[canvas is zx plane at y=6,gray,very thin] (\col*\rectWidth, 0) -- ++(0, \gridRows*\rectHeight);
    \draw[canvas is xy plane at z=6,gray,very thin] (\col*\rectWidth, 0) -- ++(0, \gridRows*\rectHeight);
  }

  % Draw horizontal lines
  \foreach \row in {0,...,\gridRows} {
    \draw[canvas is zy plane at x=6,gray,very thin] (0, \row*\rectHeight) -- ++(\gridCols*\rectWidth, 0);
    \draw[canvas is zx plane at y=6,gray,very thin] (0, \row*\rectHeight) -- ++(\gridCols*\rectWidth, 0);
    \draw[canvas is xy plane at z=6,gray,very thin] (0, \row*\rectHeight) -- ++(\gridCols*\rectWidth, 0);
  }

  % Redefine grid dimensions
  \def\gridRows{3}
  \def\gridCols{3}
  % Redefine rectangle dimensions
  \def\rectWidth{2}
  \def\rectHeight{2}

  % Draw vertical lines
  \foreach \col in {0,...,\gridCols} {
    \draw[canvas is zy plane at x=6,thick] (\col*\rectWidth, 0) -- ++(0, \gridRows*\rectHeight);
    \draw[canvas is zx plane at y=6,thick] (\col*\rectWidth, 0) -- ++(0, \gridRows*\rectHeight);
    \draw[canvas is xy plane at z=6,thick] (\col*\rectWidth, 0) -- ++(0, \gridRows*\rectHeight);
  }

  % Draw horizontal lines
  \foreach \row in {0,...,\gridRows} {
    \draw[canvas is zy plane at x=6,thick] (0, \row*\rectHeight) -- ++(\gridCols*\rectWidth, 0);
    \draw[canvas is zx plane at y=6,thick] (0, \row*\rectHeight) -- ++(\gridCols*\rectWidth, 0);
    \draw[canvas is xy plane at z=6,thick] (0, \row*\rectHeight) -- ++(\gridCols*\rectWidth, 0);
  }

\end{scope}
\end{tikzpicture}
\end{document}
'''

body = r'''
  \foreach \col in {1,...,\gridCols} {
    \foreach \row in {1,...,\gridRows} {
        \draw[canvas is zy plane at x=6, thick] node[anchor=center] at (\col*\rectWidth-0.5,\row*\rectHeight-0.5) {\scriptsize A};
    }
  }
  \foreach \col in {1,...,\gridCols} {
    \foreach \row in {1,...,\gridRows} {
        \draw[canvas is xy plane at z=6, thick] node[anchor=center] at (\col*\rectWidth-0.5,\row*\rectHeight-0.5) {\scriptsize A};
    }
  }
  \foreach \col in {1,...,\gridCols} {
    \foreach \row in {1,...,\gridRows} {
        \draw[canvas is zx plane at y=6, thick] node[anchor=center] at (\col*\rectWidth-0.5,\row*\rectHeight-0.5) {\scriptsize A};
    }
  }
'''

col = 1
row = 1
body_lines = [r'''draw[canvas is zy plane at x=6, thick] node[anchor=center] at ''' + f"({col}*{chr(92)}rectWidth-0.5,{row}*{chr(92)}rectHeight-0.5) {{{chr(92)}scriptsize A}};"]

print(header)

for l in sys.stdin:
    if l == "\n" or l.startswith("#"):
        continue
    axis, row, col, code = l.strip().split("\t")
    plane = {"x": "zy", "y": "zx", "z": "xy"}[axis]
    slant = {"x": "yslant=-0.5,xslant=0", "y": "yslant=0.5,xslant=0", "z": "yslant=0,xslant=0"}[axis]
    line = f"{chr(92)}draw[canvas is {plane} plane at {axis}=6, thick] node[anchor=center,{slant}] at ({col}*{chr(92)}rectWidth-0.5,{row}*{chr(92)}rectHeight-0.5) {{{chr(92)}scriptsize {code}}};"
    print(line)



print(footer)

'''
  \foreach \col in {1,...,\gridCols} {
    \foreach \row in {1,...,\gridRows} {
        \draw[canvas is zy plane at x=6, thick] node[anchor=center,yslant=-0.5,xslant=0] at (\col*\rectWidth-0.5,\row*\rectHeight-0.5) {\scriptsize X};
    }
  }
  \foreach \col in {1,...,\gridCols} {
    \foreach \row in {1,...,\gridRows} {
        \draw[canvas is xy plane at z=6, thick] node[anchor=center,yslant=0,xslant=0] at (\col*\rectWidth-0.5,\row*\rectHeight-0.5) {\scriptsize Z};
    }
  }
  \foreach \col in {1,...,\gridCols} {
    \foreach \row in {1,...,\gridRows} {
        \draw[canvas is zx plane at y=6, thick] node[anchor=center,yslant=0.5,xslant=0] at (\col*\rectWidth-0.5,\row*\rectHeight-0.5) {\scriptsize Y};
    }
  }
'''
