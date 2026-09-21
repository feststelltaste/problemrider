---
title: Style Guide
description: Sicherstellung konsistenten Designs und Nutzererlebnisses.
category:
- Requirements
- Code
quality_tactics_url: https://qualitytactics.de/en/usability/style-guide/
problems:
- inconsistent-behavior
- inconsistent-codebase
- poor-user-experience-ux-design
- inconsistent-coding-standards
- user-confusion
- undefined-code-style-guidelines
- mixed-coding-styles
- maintenance-overhead
- clever-code
- inconsistent-execution
- inconsistent-naming-conventions
- nitpicking-culture
- poor-naming-conventions
- style-arguments-in-code-reviews
layout: solution
lang: de
en_slug: style-guide
related_solutions:
- slug: consistent-user-interface
  similarity: 0.85
- slug: consistent-terminology
  similarity: 0.8
- slug: intuitive-navigation
  similarity: 0.8
- slug: user-centered-design
  similarity: 0.75
- slug: pattern-language
  similarity: 0.75
- slug: design-tokens
  similarity: 0.75
---

## Description

Ein Style Guide dokumentiert das visuelle Design, die Interaktionsmuster und die code-seitigen Konventionen, zu denen sich ein Team verpflichtet, als lebendige, durchsuchbare Referenz statt als statisches Dokument, das niemand konsultiert. Legacy-Systeme, die von aufeinanderfolgenden Teams über Jahre gebaut wurden, driften zuverlässig in einen Zustand, in dem die Schaltflächen des Finanzmoduls nichts mit denen des HR-Moduls gemein haben, weil niemand jemals von einem gemeinsamen Standard aus arbeitete — der Style Guide existiert speziell, um diese Fragmentierung davon abzuhalten fortzuschreiten, auch wenn er nicht rückwirkend alles bereits Gebaute reparieren kann. Er verdient sich seinen Nutzen nur, wenn er echt durch Code-Review durchgesetzt und aktuell gehalten wird, während neue Muster entstehen, da ein Style Guide, der Konventionen beschreibt, über die die Codebasis bereits hinausgewachsen ist, genau zu der Art ignoriertem Artefakt wird, das er verhindern sollte.

## How to Apply ◆

> Legacy-Systeme, die über viele Jahre von mehreren Teams entwickelt wurden, sammeln Inkonsistenzen im visuellen Design, in Interaktionsmustern und Code-Konventionen an. Ein Style Guide etabliert Standards, die weitere Fragmentierung verhindern.

- Erstellen Sie ein lebendiges Style-Guide-Dokument, das visuelle Designelemente einschließlich Farben, Typografie, Abstände, Icons und Komponentenspezifikationen abdeckt. Beziehen Sie Do- und Don't-Beispiele ein.
- Definieren Sie Interaktionsmuster für gängige UI-Aufgaben wie CRUD-Operationen, Suche und Filter, Paginierung und Benachrichtigungen. Dokumentieren Sie, wann Modals gegenüber Inline-Bearbeitung genutzt werden sollen und wie Ladezustände gehandhabt werden.
- Beziehen Sie code-seitige Standards für die Frontend-Entwicklung ein: Komponenten-Namenskonventionen, CSS-Methodik, State-Management-Muster und Barrierefreiheitsanforderungen.
- Bauen Sie den Style Guide als durchsuchbare Referenz mit lebenden Komponentenbeispielen, nicht als statisches PDF-Dokument. Entwickler sollten die standardisierten Komponenten sehen und mit ihnen interagieren können.
- Setzen Sie Einhaltung durch Code-Review durch. Beziehen Sie Style-Guide-Konformität als Checklisten-Element in Pull-Request-Reviews für jede UI-Änderung ein.
- Aktualisieren Sie den Style Guide, wenn neue Muster etabliert werden, und entfernen Sie Muster, die nicht mehr genutzt werden. Ein veralteter Style Guide wird ignoriert.

## Tradeoffs ⇄

> Ein Style Guide verhindert die weitere Ansammlung von Inkonsistenz, erfordert aber Investition zur Erstellung und Disziplin zur Durchsetzung.

**Vorteile:**

- Verhindert die fortgesetzte Ansammlung visueller und verhaltensbezogener Inkonsistenzen, während sich das Legacy-System weiterentwickelt.
- Beschleunigt die Frontend-Entwicklung durch bereitgestellte, einsatzbereite Muster und Komponenten, statt jeden Entwickler von Grund auf gestalten zu lassen.
- Reduziert die kognitive Last für Nutzer, indem sichergestellt wird, dass ähnliche Aufgaben in der gesamten Anwendung gleich aussehen und sich gleich verhalten.
- Bietet ein gemeinsames Vokabular für die Diskussion von Designentscheidungen und reduziert subjektive Debatten während Code-Reviews.

**Kosten und Risiken:**

- Die Erstellung eines umfassenden Style Guides erfordert erheblichen Vorabaufwand sowohl aus Design- als auch aus Entwicklungsperspektive.
- Ein zu starrer Style Guide kann Innovation ersticken und Teams davon abhalten, mit besseren Interaktionsmustern zu experimentieren.
- Die Durchsetzung der Style-Guide-Konformität in einem großen Team, das an einem Legacy-System arbeitet, erfordert Wachsamkeit, da Legacy-Code, der dem Guide vorausgeht, ihn weiterhin verletzen wird, bis er refaktoriert ist.
- Die Pflege des Style Guides als lebendes Dokument erfordert laufenden Aufwand; ein aufgegebener Style Guide wird schnell zu einem irreführenden Artefakt.

## How It Could Be

> Ohne einen Style Guide trifft jeder Entwickler unabhängige Designentscheidungen, was zu einem System führt, das sich wie mehrere zusammengeflickte Anwendungen anfühlt.

Ein Legacy-Enterprise-Resource-Planning-System hat in fast jedem Modul unterschiedliche visuelle Stile, weil jedes von einem anderen Team im Laufe des letzten Jahrzehnts gebaut wurde. Schaltflächen im Finanzmodul sind blau und rechteckig, im HR-Modul sind sie grün und abgerundet, und im Inventarmodul sind sie grau mit reiner Textgestaltung. Nutzer, die modulübergreifend arbeiten, finden die Inkonsistenz desorientierend. Das Team erstellt einen Style Guide, der alle interaktiven Elemente standardisiert, beginnend mit Schaltflächen, Formularen und Tabellen. Sie bauen eine Komponentenbibliothek, die den Style Guide implementiert, und verlangen, dass aller neuer Code und bedeutende Modifikationen Komponenten aus der Bibliothek nutzen. Im Laufe eines Jahres erreichen die am stärksten modifizierten Module visuelle Konsistenz, und das Nutzerfeedback wechselt von Beschwerden über verwirrende Inkonsistenzen zu Wertschätzung für die polierteren Erfahrung.
