---
title: Duplikaterkennung
description: Systematisches Auffinden, wo Code kopiert wurde, und Prüfung, ob die
  Kopien auseinandergedriftet sind — denn die gefährlichen Duplikate sind die, von
  denen niemand weiß.
category:
- Code
- Testing
- Process
problems:
- code-duplication
- copy-paste-programming
- partial-bug-fixes
- regression-bugs
- inconsistent-execution
- high-technical-debt
- increased-bug-count
- maintenance-cost-increase
- difficult-to-understand-code
- brittle-codebase
- quality-degradation
- hidden-dependencies
- large-estimates-for-small-changes
- low-code-customization-sprawl
layout: solution
lang: de
en_slug: duplication-detection
related_solutions:
- slug: code-hotspot-analysis
  similarity: 0.7
- slug: static-analysis-and-linting
  similarity: 0.7
- slug: code-metrics
  similarity: 0.65
- slug: code-reading-sessions
  similarity: 0.65
- slug: technical-debt-assessment
  similarity: 0.65
- slug: code-review-process-reform
  similarity: 0.65
---

## Description

Duplikaterkennung identifiziert systematisch Codepassagen, die im Wesentlichen gleich sind, indem normalisierte Struktur statt rohem Text verglichen wird, sodass umbenannte Variablen und neu formatierter Leerraum eine Kopie nicht verstecken. Ihr Wert in Legacy-Systemen liegt nicht in der gemeldeten Gesamtzahl, die als Qualitätsmaß nahezu nutzlos ist. Er liegt in der Entdeckung spezifischer Kopien, an die sich niemand erinnerte, sie gemacht zu haben. Ein in einer Kopie behobener und in den anderen nicht behobener Defekt ist eines der häufigsten und frustrierendsten Legacy-Fehlermuster: Der Fehler tritt wieder auf, wird als neu untersucht und wird in einer anderen Kopie erneut behoben, manchmal über Jahre. Erkennung verwandelt das von einer unangenehmen Überraschung in eine überprüfbare Tatsache — wenn Sie etwas ändern, können Sie fragen, ob diese Logik anderswo existiert, und eine Antwort erhalten.

## How to Apply ◆

> Die Duplizierung, die zählt, ist nicht die Passage, von der jeder weiß, dass sie kopiert wurde; es ist die, die von jemandem kopiert wurde, der 2016 ging, in einem Modul, das niemand mit diesem hier verbindet.

- **Vergleichen Sie Struktur, nicht Text.** Erkennung, die auf rohen Zeichen arbeitet, verpasst alles, wo eine Variable umbenannt wurde oder die Formatierung abweicht, was in der Praxis das meiste ist. Bezeichner und Layout vor dem Vergleich zu normalisieren ist das, was die Ergebnisse lesenswert macht.
- **Erkennen Sie, dass Klone in Graden kommen**: identische Passagen, Passagen, die sich nur in Namen unterscheiden, umstrukturierte Passagen unter Beibehaltung des Verhaltens, und Passagen, die dasselbe anders geschrieben tun. Erkennung findet zuverlässig die ersten drei; die vierte liegt generell außerhalb ihrer Reichweite und braucht menschliches Lesen.
- **Ignorieren Sie die Schlagzeilenprozentzahl.** „Diese Codebasis ist zu 14 Prozent dupliziert" ist eine Zahl ohne angehängte Entscheidung. Die nützliche Ausgabe ist eine Liste spezifischer Duplikatgruppen, und die Summe ist hauptsächlich für einen Trend gut.
- **Priorisieren Sie die abgedrifteten Kopien.** Zwei noch identische Kopien sind eine Wartungskosten; zwei auseinandergedriftete Kopien sind ein latenter Defekt, weil jemand bereits eine geändert hat und die andere nicht. Abgedriftete Gruppen sind die wertvollsten Befunde und sollten zuerst überprüft werden.
- **Priorisieren Sie die Kopien in sich änderndem Code.** Duplizierung in einem Modul, das nichts anfasst, kostet nichts. Die Erkennungsergebnisse mit der Änderungshäufigkeit zu kreuzen reduziert eine Liste von Hunderten auf eine Handvoll, die eine Handlung wert ist.
- **Suchen Sie nach Duplizierung, die Eigentümerschaftsgrenzen überschreitet.** Zwei Teams, die dieselbe Logik unabhängig pflegen, ist ein organisatorischer Befund ebenso sehr wie ein technischer, und es bedeutet meist, dass ein gemeinsames Konzept nie benannt oder mit einem Eigentümer versehen wurde.
- **Nutzen Sie es als Prüfung vor der Behebung eines Defekts**, nicht nur als periodischen Bericht. „Existiert diese Logik irgendwo sonst" ist die Frage, die eine partielle Korrektur verhindert, und sie zu beantworten dauert Sekunden, sobald Erkennung verfügbar ist.
- **Entfernen Sie nicht jedes Duplikat.** Zwei Passagen, die heute aus unverwandten Gründen ähnlich sind, werden morgen auseinanderdriften, und sie zusammenzuführen schafft eine Kopplung, die schlimmer ist als die Duplizierung. Bewusste Duplizierung über separate Geschäftskontexte hinweg ist häufig das korrekte Design.
- **Schließen Sie aus, was ausgeschlossen werden sollte**: generierter Code, vendorisierte Abhängigkeiten und Test-Fixtures, bei denen Wiederholung die Lesbarkeit unterstützt. Erkennung, die diese meldet, wird komplett ignoriert, einschließlich der Befunde, die zählten.
- **Verfolgen Sie den Trend und paaren Sie ihn mit einer Sperrklinke**, sodass sich keine neue Duplizierung anhäuft, während alte entfernt wird.

## Tradeoffs ⇄

> Erkennung findet Kopien, von denen niemand wusste, und verwandelt partielle Korrekturen in eine verhinderbare Defektklasse, aber die rohe Ausgabe ist rauschbehaftet, und Duplizierung zu entfernen ist nicht immer eine Verbesserung.

**Vorteile:**

- Unbekannte Kopien werden gefunden, was der einzige Weg ist, das Muster zu stoppen, bei dem ein Defekt an einer Stelle behoben wird und von einer anderen wieder auftritt.
- Abgedriftete Kopien tauchen als konkrete Befunde auf, und Divergenz ist direkter Beleg dafür, dass eine Änderung bereits inkonsistent angewandt wurde.
- Die Prüfung vor einer Defektbehebung ist günstig und verhindert partielle Korrekturen, was wahrscheinlich der größte praktische Nutzen ist.
- Duplizierung über Teamgrenzen hinweg offenbart fehlende gemeinsame Konzepte und Eigentümerschaftslücken, die keine andere Analyse aufdeckt.
- Der Trend gibt ein objektives Signal darüber, ob sich die Copy-Paste-Praxis verbessert, was sonst eine Frage des Eindrucks ist.

**Kosten und Risiken:**

- Die rohe Ausgabe ist rauschbehaftet und von Befunden dominiert, die nicht zählen, sodass sie nach Änderungshäufigkeit und Divergenz gefiltert werden muss, um überhaupt nutzbar zu sein.
- Sie misst textuelle und strukturelle Ähnlichkeit, nicht konzeptuelle Duplizierung, sodass sie Logik verpasst, die neu implementiert statt kopiert wurde — oft die schädlichere Art.
- Die Duplizierungsprozentzahl als Qualitätsziel zu behandeln lädt dazu ein, Duplizierung zu entfernen, die man hätte lassen sollen, was verfrühte Abstraktionen erzeugt, die unverwandte Dinge koppeln.
- Das Zusammenführen von Duplikaten schafft Kopplung, und zufällig ähnliche Passagen werden später auseinanderdriften, an welchem Punkt die gemeinsame Abstraktion zu einem Hindernis wird.
- Die Konfiguration der Ausschlüsse und Schwellenwerte braucht Iteration, und ein unkonfigurierter Lauf erzeugt einen Bericht, der die Praxis diskreditiert.

## How It Could Be

Ein Team hatte über zwei Jahre denselben Rundungsdefekt in einer Rechnungsberechnung dreimal behoben, jedes Mal als neu gemeldeter Fehler, jedes Mal in einer anderen Datei. Erkennung über ihre Codebasis auszuführen fand die Berechnung an fünf Stellen, von denen vier auf kleine Weise voneinander abgedriftet waren — eine hatte die Korrektur, zwei hatten unterschiedliche partielle Korrekturen, und eine war seit ihrer Kopie 2014 nie angefasst worden. Diese fünfte Kopie war in einem Batch-Job, der einen monatlichen Bericht erzeugte, den ein Finanzteam jeden Monat manuell abglich, ein manueller Schritt, dessen Ursprung niemand erklären konnte. Die fünf zu einer Implementierung zu konsolidieren dauerte eine Woche, und der monatliche Abgleich hörte auf.

Die dauerhaftere Änderung des Teams war prozedural statt behebend. Sie fügten ihrer Defektbehebungsroutine eine einzige Frage hinzu: Prüfen Sie vor der Behebung, ob diese Logik anderswo erscheint. Im folgenden Jahr fand diese Prüfung Duplikate in 11 von etwa 90 Defektbehebungen, und in 4 dieser Fälle musste die Korrektur an mehr als einer Stelle angewandt werden. Ihr früherer Versuch, Duplizierung anzugehen — eine Zielprozentzahl im Build — war aufgegeben worden, weil sich die Zahl nur verbessern ließ, indem zufällig ähnliche Passagen zusammengeführt wurden, und zwei dieser Zusammenführungen hatten später rückgängig gemacht werden müssen. Die Schlussfolgerung des Teams war, dass Duplikaterkennung als Nachschlagewerkzeug wertvoll und als Ziel wertlos war.
