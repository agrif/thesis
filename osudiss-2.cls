%%
%% This is file `osudiss-2.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% osudiss-2.dtx  (with options: `class')
%% 
%% This is a generated file.
%% 
%% Copyright (C) 2010  by Steven Avery <avery@mps.ohio-state.edu>
%% 
%% This file may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either
%% version 1.2 of this license or (at your option) any later
%% version. The latest version of this license is in:
%% 
%%    http://www.latex-project.org/lppl.txt
%% 
%% and version 1.2 or later is part of all distributions of
%% LaTeX version 1999/12/01 or later.
%% 
\def\filename{osudiss-2}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{osudiss-2}
    [2010/05/24 v0.9 .dtx osudiss-2 file]
\typeout{*** Conforms to the 2010 Graduate School Format ***}
%% \changes{v1.0}{2010/05/04}{Initial version}
\RequirePackage{ifthen}
\newboolean{@footbychap} % option
\newboolean{@footrule}   % option
\newboolean{@draft}      % option
\newboolean{@phd}        % option
\newboolean{@masters}    % option
\newboolean{@hyper}      % option
\newboolean{@advisor}    % determine if \advisorname has been called yet
\newboolean{@coadvisor}  % determine if \coadvisorname has been called yet
\newboolean{@member}     % determine if \member has been called yet
\setboolean{@footbychap}{false}
\setboolean{@footrule}{false}
\setboolean{@draft}{false}
\setboolean{@phd}{true}
\setboolean{@masters}{false}
\setboolean{@hyper}{true}
\setboolean{@advisor}{false}
\setboolean{@coadvisor}{false}
\setboolean{@member}{false}
\DeclareOption{a4paper}{
  \ClassWarningNoLine{\filename}
   {`a4paper' option not GS-compliant.
    \MessageBreak Option will be ignored}}
\DeclareOption{a5paper}{
  \ClassWarningNoLine{\filename}
   {`a5paper' option not GS-compliant.
    \MessageBreak Option will be ignored}}
\DeclareOption{b5paper}{
  \ClassWarningNoLine{\filename}
   {`b5paper' option not GS-compliant.
    \MessageBreak Option will be ignored}}
\DeclareOption{legalpaper}{
  \ClassWarningNoLine{\filename}
   {`legalpaper' option not GS-compliant.
    \MessageBreak Option will be ignored}}
\DeclareOption{executivepaper}{
  \ClassWarningNoLine{\filename}
   {`executivepaper' option not GS-compliant.
    \MessageBreak Option will be ignored}}
\DeclareOption{notitlepage}{
  \ClassWarningNoLine{\filename}
   {`notitlepage' option not GS-compliant and not supported.
    \MessageBreak Option will be ignored}}
\DeclareOption{twocolumn}{
  \ClassWarningNoLine{\filename}
   {`twocolumn' option not GS-compliant and not supported.
    \MessageBreak Option will be ignored}}
\DeclareOption{landscape}{
  \ClassWarningNoLine{\filename}
   {`landscape' option not GS-compliant.
    \MessageBreak Option will be ignored}}
\DeclareOption{openright}{
  \ClassWarningNoLine{\filename}
   {`openright' option not GS-compliant}}
\DeclareOption{twoside}{
  \ClassWarningNoLine{\filename}
   {`twoside' option not GS-compliant.
    \MessageBreak Option will be ignored}}
\DeclareOption{10pt}{\PassOptionsToClass{\CurrentOption}{report}}
\DeclareOption{11pt}{\PassOptionsToClass{\CurrentOption}{report}}
\DeclareOption{12pt}{\PassOptionsToClass{\CurrentOption}{report}}
\DeclareOption{openbib}{\PassOptionsToClass{\CurrentOption}{report}}
\DeclareOption{final}{\PassOptionsToClass{\CurrentOption}{report}}
\DeclareOption{leqno}{\PassOptionsToClass{\CurrentOption}{report}}
\DeclareOption{fleqn}{\PassOptionsToClass{\CurrentOption}{report}}
\newcommand\thespacing{\doublespacing} % Default
\newcommand\thespace{doublespace}      % Default
\DeclareOption{single}{
  \renewcommand\thespacing\singlespacing
  \renewcommand\thespace{singlespace}
  \ClassWarningNoLine{\filename}{`single' option not GS-compliant}}
\DeclareOption{onehalf}{
  \renewcommand\thespacing\onehalfspacing
  \renewcommand\thespace{onehalfspace}}
\DeclareOption{double}{
  \renewcommand\thespacing\doublespacing
  \renewcommand\thespace{doublespace}}
\DeclareOption{hyper}{
  \setboolean{@hyper}{true}} % Default
\DeclareOption{nohyper}{
  \setboolean{@hyper}{false}}
\DeclareOption{footbychap}{\setboolean{\@footbychap}{true}}
\DeclareOption{nofootreset}{\setboolean{\@footbychap}{false}} % Default
\DeclareOption{footrule}{\setboolean{\@footrule}{true}}
\DeclareOption{nofootrule}{\setboolean{@footrule}{false}} % Default
\DeclareOption{draft}{\setboolean{@draft}{true}\ClassWarningNoLine{\filename}
                                                {`draft' option not GS-compliant}
                      \PassOptionsToClass{\CurrentOption}{report}}
\DeclareOption{masters}{\setboolean{@masters}{true}\setboolean{@phd}{false}}
\DeclareOption{phd}{\setboolean{@masters}{false}\setboolean{@phd}{true}} % Default
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ExecuteOptions{11pt, onecolumn, letterpaper, oneside}
\ProcessOptions\relax
\LoadClass{report}
\RequirePackage{geometry}  % easier control of page layout
\RequirePackage[nodisplayskipstretch]{setspace}  % easier control of line spacing
\RequirePackage{titlesec}  % easier control of section formatting
\RequirePackage{tocloft}   % better control over ToC, LoF, and LoT
\ifthenelse{\boolean{@hyper}} % fixes pdf page numbering}
{
  \RequirePackage[pdfpagelabels, final, implicit=true]{hyperref}
}
{
  \RequirePackage[pdfpagelabels, draft, implicit=true]{hyperref}
}
\RequirePackage{calc}
\ifthenelse{\boolean{@footbychap}}
{}
{
  \RequirePackage{remreset}
  \@removefromreset{footnote}{chapter}
}
\ifthenelse{\boolean{@draft}}
{
  \RequirePackage{sa-draftwater}
}{}
\ifthenelse{\boolean{@footrule}}{}
{
  \renewcommand{\footnoterule}{}
}
\newcommand\typesetColumnHeading[1]{\textbf{#1}}
\newcommand\typesetLevelOne[1]{{\normalfont\Huge\textsc{#1}\par}}
\newcommand\typesetLevelTwo[1]{{\normalfont\Huge\textbf{#1}}}
\newcommand\typesetLevelThree[1]{{\normalfont\Large\textbf{#1}}}
\newcommand\typesetLevelFour[1]{{\normalfont\large\textbf{#1}}}
\newcommand\typesetLevelFive[1]{{\normalfont\normalsize\textbf{#1}}}
\newcommand\titleFont{\normalfont\Huge\sc}
\newcommand\titlePageFont{\normalfont\large}
\renewcommand\contentsname{Table of Contents}
\renewcommand\partname{Volume}
\renewcommand\listtablename{List of Tables}
\renewcommand\listfigurename{List of Figures}
\renewcommand\abstractname{Abstract}
\renewcommand\appendixname{Appendix}
\renewcommand\chaptername{Chapter}
\renewcommand\bibname{Bibliography}
\newcommand\pagename{Page}
\newcommand\chaptersname{Chapters}
\newcommand\appendicesname{Appendices}
\newcommand\acknowledgmentsname{Acknowledgments}
\newcommand\advisorword{Advisor}
\newcommand\coadvisorword{Co-Advisor}
\newcommand\fieldsname{Fields of Study}
\newcommand\publicationsname{Publications}
\newcommand\vitaname{Vita}
\newcommand\dedicationname{Dedication}
\newcommand\majorfieldname{Major Field}
\newcommand\@unit{Graduate Program in Physics}        % default
\newcommand\@graduationyear{\number\year}             % default
\ifthenelse{\boolean{@phd}}
{
  \newcommand\@degree{Doctor of Philosophy} % default
  \newcommand\@thesis{Dissertation}
  \newcommand\@committee{Dissertation}
}
{
  \ifthenelse{\boolean{@masters}}
  {
      \newcommand\@degree{Masters of Science} % default
      \newcommand\@thesis{A Thesis}
      \newcommand\@committee{Master's Examination}
  }{\relax}
}
\geometry{top=1in, left=1.5in, right=1in, bottom=1in, nohead, includefoot,
          footskip=1.5\baselineskip, letterpaper, portrait,
          twocolumn=false, twoside=false}
%%\pdfpagewidth \paperwidth
%%\pdfpageheight \paperheight
\newcommand\@advisorlist{\relax}
\newcommand\advisorname[1]{
  \ifthenelse{\boolean{@advisor}}
  {
    \protected@xdef\@advisorlist{\@advisorlist\\ #1, \advisorword}
  }
  {
    \setboolean{@advisor}{true}
    \protected@xdef\@advisorlist{#1, \advisorword}
  }
}
\newcommand\@coadvisorlist{\relax}
\newcommand\coadvisorname[1]{
  \ifthenelse{\boolean{@coadvisor}}
  {
    \protected@xdef\@coadvisorlist{\@coadvisorlist\\ #1, \coadvisorword}
  }
  {
    \setboolean{@coadvisor}{true}\protected@xdef\@coadvisorlist{#1, \coadvisorword}
  }
}
\newcommand\@memberlist{\relax}
\newcommand\member[1]{
  \ifthenelse{\boolean{@member}}
  {
    \protected@xdef\@memberlist{\@memberlist\\#1}
  }
  {
    \setboolean{@member}{true}\protected@xdef\@memberlist{#1}
  }
}
\newcommand\unit[1]{\gdef\@unit{#1}}
\newcommand\graduationyear[1]{\gdef\@graduationyear{#1}}
\newcommand\@authordegreees{\relax}
\newcommand\authordegrees[1]{\gdef\@authordegrees{#1}}
\newcommand\degree[1]{\gdef\@degree{#1}}
\newlength\titleskip
\setlength\titleskip{1in-\topskip-\parskip-\lineskip-\baselineskip}
\renewcommand{\maketitle}{
  \begin{titlepage}
  \centering\singlespacing
  \vspace*{\titleskip}
  \parbox{\textwidth}{\centering\singlespacing\titleFont\@title\par}
  \titlePageFont

  \vspace{3\baselineskip}
  \MakeUppercase{\@thesis}

  \vspace{3\baselineskip}
  {Presented in Partial Fulfillment of the
   Requirements for the Degree \@degree{} in the Graduate School of
   The Ohio State University}

  \vspace{1\baselineskip}
  \begin{doublespace}By\\ {\@author, \@authordegrees}\\ \@unit\end{doublespace}

  \vspace{2\baselineskip}
  \begin{doublespace}The Ohio State University\\ \@graduationyear\end{doublespace}

  \vfill
  \begin{doublespace}
  \@committee{} Committee:\normalsize\\
  \ifthenelse{\boolean{@coadvisor}}{\@advisorlist\\ \@coadvisorlist}{\@advisorlist}\\
  \@memberlist
  \end{doublespace}
  \vfill
  \end{titlepage}
}
\newlength\chapterskip
\setlength\chapterskip{1in-\topskip-\parskip-\lineskip-\baselineskip-4pt}
\titleformat{\chapter}[display]{\centering\singlespacing}
                               {\typesetLevelTwo{\chaptertitlename{} \thechapter}}
                               {.5em}
                               {\typesetLevelOne}
\titlespacing*{\chapter}{0ex}{\chapterskip}{3em}
\titleformat{\section}[hang]{}{\typesetLevelThree{\thetitle}}{1em}{\typesetLevelThree}
\titleformat{\subsection}[hang]{}{\typesetLevelFour{\thetitle}}{1em}{\typesetLevelFour}
\titleformat{\subsubsection}[hang]{}{}{}{\typesetLevelFive}
\newcommand\oldpart{This will give an error if
                    \textbackslash oldpart already exists.}
\let\oldpart\part
\renewcommand\part{\ClassWarning{\filename}{parts may not be GS-compliant
                                            and is not fully supported}\oldpart}
\newcommand\oldappendix{This will give an error if
                        \textbackslash oldappendix already exists.}
\let\oldappendix\appendix
\renewcommand\appendix{\oldappendix
       %manually add \chaptersname to table of contents
       \addtocontents{toc}{\protect \contentsline {part}{\appendicesname}{\pageref{appendix.A}}{appendix.A}}
       \clearpage
       \pdfbookmark[-1]{\appendicesname}{appendices} %manually add PDF bookmark for Chapters
       }
\newcommand\copyrightpage{
  \clearpage
  \thispagestyle{empty}
  \null\vfill
    \begin{center}\begin{doublespace}
      \copyright \ Copyright by\\
      \@author\\
      {\@graduationyear}
    \end{doublespace}\end{center}
  \vfill
  \ifthenelse{\boolean{@draft}}
  {
    \draftwaterpage
  }{}
  \pagebreak
}
\providecommand*{\toclevel@front}{0} %define a table of contents level for type front, to prevent warnings

\renewenvironment{abstract}{
  \clearpage
  \setlength\chapterskip{1in-\topskip-\parskip-\lineskip-\baselineskip-1em}
  \chapter*{\abstractname}
  \addcontentsline{toc}{front}{\abstractname}
  \thispagestyle{plain}\begin{\thespace}
}
{
  \end{\thespace}
  \setlength\chapterskip{1in-\topskip-\parskip-\lineskip-\baselineskip-4pt}
  \pagebreak
}
\newcommand\dedication[1]{
  \clearpage
  \thispagestyle{plain}
  \phantomsection %add a phantom section to allow the proper page number to appear in the TOC and proper hyperlinking
  \addcontentsline{toc}{front}{\dedicationname}
  \null\vfill
  \begin{center}#1\end{center}
  \vfill\pagebreak}
\newenvironment{acknowledgments}{
  \clearpage
  \setlength\chapterskip{1in-\topskip-\parskip-\lineskip-\baselineskip-1em}
  \chapter*{\acknowledgmentsname}
  \addcontentsline{toc}{front}{\acknowledgmentsname}
  \thispagestyle{plain}\begin{\thespace}
}
{
  \end{\thespace}
  \setlength\chapterskip{1in-\topskip-\parskip-\lineskip-\baselineskip-4pt}
  \pagebreak
}
\let\acknowledgements\acknowledgments
\let\endacknowledgements\endacknowledgments
\newenvironment{vita}{
  \clearpage
  \setlength\chapterskip{1in-\topskip-\parskip-\lineskip-\baselineskip-1em}
  \chapter*{\vitaname}
  \addcontentsline{toc}{front}{\vitaname}
  \thispagestyle{plain}\singlespacing
}
{
  \setlength\chapterskip{1in-\topskip-\parskip-\lineskip-\baselineskip-4pt}
  \pagebreak
}
\newcommand\dateitem[2]{\noindent #1 \dotfill \parbox[t]{2.75in}{#2}\par~\par}
\newenvironment{publist}
  {\vspace*{15pt}\begin{center}{{\typesetLevelThree\publicationsname}}
   \end{center}\vspace*{10pt}}
  {\vspace{\baselineskip}}
\newcommand\pubitem[1]{\noindent #1 \par~~\par}
\newenvironment{fieldsstudy}
  {\begin{center}{{\typesetLevelThree\fieldsname}}\end{center}
   \vspace*{15pt}}
  {}
\newcommand\majorfield[1]{\noindent \majorfieldname:  #1\par~\par}
\newcommand\onestudy[2]{Studies in #1: #2}
\newenvironment{studieslist}{Studies in:\par\vspace*{8pt}\hspace*{2em}\begin{tabular}{l l}}
{\end{tabular}}
\newcommand\studyitem[2]{#1 & #2\\}
\newenvironment{oldverse}{This will give an error if
                          \textbackslash oldverse already exists.}{}
\let\oldverse\verse
\let\endoldverse\endverse
\renewenvironment{verse}{\begin{oldverse}\begin{singlespace}}
                        {\end{singlespace}\end{oldverse}}
\newenvironment{oldquotation}{This will give an error if
                              \textbackslash oldquotation already exists.}{}
\let\oldquotation\quotation
\let\endoldquotation\endquotation
\renewenvironment{quotation}{\begin{oldquotation}\begin{singlespace}}
                            {\end{singlespace}\end{oldquotation}}
\newenvironment{oldquote}{This will give an error if
                          \textbackslash oldquote already exists.}{}
\let\oldquote\quote
\let\endoldquote\endquote
\renewenvironment{quote}{\begin{oldquote}\begin{singlespace}}
                        {\end{singlespace}\end{oldquote}}

\renewcommand\cfttoctitlefont{\hfil\typesetLevelTwo}
\renewcommand\cftaftertoctitle{\hfill}
\renewcommand\cftloftitlefont{
       \phantomsection %required to properly locate list of figures in TOC and hyperlinks
       \hfill\typesetLevelTwo}
\renewcommand\cftafterloftitle{
       \hfill
               \addcontentsline{toc}{front}{\typesetColumnHeading{\listfigurename}}}
\renewcommand\cftlottitlefont{
       \phantomsection %required to properly locate list of tables in TOC and hyperlinks
       \hfill\typesetLevelTwo}
\renewcommand\cftafterlottitle{
               \hfill
               \addcontentsline{toc}{front}{\typesetColumnHeading{\listtablename}}}
\cftpagenumbersoff{part}
\setlength\cftbeforetoctitleskip{1in-\topskip-1pt}
\setlength\cftbeforeloftitleskip{1in-\topskip-1pt}
\setlength\cftbeforelottitleskip{1in-\topskip-1pt}
\newcommand*\l@front{\@dottedtocline{0}{0pt}{2.3em}}
\addtocontents{lof}{
    \vskip 0.55in
    \typesetColumnHeading{\figurename}\hfill\typesetColumnHeading{\pagename}
    \vskip 0.25em
    \rm
}
\addtocontents{lot}{
    \vskip 0.55in
    \typesetColumnHeading{\tablename}\hfill\typesetColumnHeading{\pagename}
    \vskip 0.25em
    \rm
}
\addtocontents{toc}{~\hfill{\typesetColumnHeading{\pagename}}}
\newenvironment{oldthebibliography}{This will give an error if
               \textbackslash oldthebibliography already exists.}{}
\let\oldthebibliography\thebibliography
\let\endoldthebibliography\endthebibliography
\renewenvironment{thebibliography}[1]
                 {\begin{oldthebibliography}{#1}\begin{singlespace}}
                 {\end{singlespace}\end{oldthebibliography}}
\newcommand\frontmatter{\pagenumbering{alph} %change (undisplayed) page numbering of title page to 'a' to prevent warning about duplicate page numbers
\maketitle\pagenumbering{roman}\copyrightpage}
\newcommand\mainmatter{\addtocontents{toc}{\protect \contentsline {part}{\chaptersname}{\pageref{chapter.1}}{chapter.1}} %manually add a TOC entry pointing to the first chapter
                       \clearpage
       \pdfbookmark[-1]{\chaptersname}{chapters} %manually add a PDF bookmark pointing to the first chapter
       \pagenumbering{arabic}\thespacing}
\newcommand\backmatter{\clearpage
\phantomsection %create a phantom section for hyperlinks to point to
\pdfbookmark[-1]{\bibname}{bib} %add PDF bookmark for the bibliography
\addtocontents{toc}{\protect \contentsline {chapter}{\bibname}{\thepage}{\@currentHref}}} %add TOC line for the bibliograpy
\setlength{\floatsep}{3.3em plus .2em minus .3em}
\setlength{\textfloatsep}{3.1em plus .2em minus .1em}
\setlength{\intextsep}{3.1em plus .2em minus .1em}
%% \dbltextfloatsep % irrelevant for single-column layout
%% \dblfloatsep     % irrelevant for single-column layout
%%\widowpenalty 10000  % completely forbids!
%%\clubpenalty 10000   % completely forbids!
%%\brokenpenalty=10000 % completely forbids!
\AtBeginDocument{
\hypersetup{pdftitle=\@title,
            pdfauthor=\@author,
            pdfsubject={\@thesis, \@unit}}
\pagestyle{plain}

\let\Oldtableofcontents\tableofcontents
\renewcommand\tableofcontents{
\pdfbookmark[0]{\contentsname}{tableofcontents}
\Oldtableofcontents}
}
\AtEndDocument{}
\AtEndOfClass{}
\AtBeginDvi{}
\endinput
%%
%% End of file `osudiss-2.cls'.
