---
title: Automatisierte Code-Migration
description: Ausdruck einer repetitiven Codeänderung als ausführbares Rezept, das
  den Syntaxbaum umschreibt, sodass eine Migration über Tausende von Aufrufstellen
  überprüfbar und wiederholbar wird.
category:
- Code
- Dependencies
- Process
problems:
- dependency-version-conflicts
- obsolete-technologies
- technology-lock-in
- api-versioning-conflicts
- legacy-api-versioning-nightmare
- code-duplication
- copy-paste-programming
- inconsistent-naming-conventions
- mixed-coding-styles
- high-technical-debt
- large-estimates-for-small-changes
- maintenance-paralysis
- increasing-brittleness
- vendor-dependency-entrapment
- fear-of-breaking-changes
- inconsistent-execution
- maintenance-cost-increase
- monolithic-functions-and-classes
- over-reliance-on-utility-classes
- refactoring-avoidance
- technology-stack-fragmentation
- undefined-code-style-guidelines
layout: solution
lang: de
en_slug: automated-code-migration
related_solutions:
- slug: large-scale-refactoring
  similarity: 0.75
- slug: continuous-dependency-updates
  similarity: 0.7
- slug: code-review-process-reform
  similarity: 0.7
- slug: code-generation
  similarity: 0.7
- slug: static-analysis-and-linting
  similarity: 0.7
- slug: small-change-batches
  similarity: 0.7
---

## Description

Automatisierte Code-Migration drückt eine repetitive Quellcodeänderung als ausführbares Rezept aus, das auf dem geparsten Syntaxbaum statt auf Text operiert, und wendet sie über eine gesamte Codebasis an. Werkzeuge dieser Art — OpenRewrite für JVM-Sprachen, Rector für PHP, jscodeshift und ts-morph für JavaScript und TypeScript, und die in die meisten IDEs eingebauten Refaktorierungs-Engines — verstehen Typen, Imports und Scope, sodass sie eine Methode über 4.000 Aufrufstellen umbenennen, eine veraltete API durch ihren Nachfolger ersetzen oder das Konfigurationsidiom eines Frameworks migrieren können, ohne die falschen Treffer, die Suchen-und-Ersetzen produziert. Die Praxis ist für Legacy-Arbeit wichtig, weil die dominante Kostenquelle eines Bibliotheks- oder Framework-Upgrades selten das Upgrade selbst ist. Es ist die mechanische Anpassung Tausender Aufrufstellen an eine geänderte API, was zu groß ist, um von Hand zu tun, zu fehleranfällig, um es mit regulären Ausdrücken zu tun, und daher überhaupt nicht getan wird — was ist, wie eine Codebasis fünf Hauptversionen im Rückstand endet.

Dies ist etwas anderes als ein Abhängigkeits-Update-Bot. Renovate und Dependabot heben die Versionsnummer an und öffnen den Pull Request; sie berühren Ihren Quellcode nicht, sodass der Build dann bei jeder Aufrufstelle scheitert, die die neue Version geändert hat. Automatisierte Code-Migration ist das, was diese Aufrufstellen behebt. Die beiden ergänzen einander: Der Bot zeigt auf, dass ein Upgrade verfügbar ist, und das Rezept macht es anwendbar.

## How to Apply ◆

> Die Upgrades, die nie geschehen, sind üblicherweise nicht die schwierigen; es sind die, die einzeln trivial sind und viertausendmal wiederholt werden.

- **Prüfen Sie auf ein bestehendes Rezept, bevor Sie eines schreiben.** Die großen Migrationen — Framework-Hauptversionen, JUnit 4 zu 5, Java-Sprachlevel-Upgrades, gängige Bibliotheksnachfolgen — sind bereits veröffentlicht, und das Ausführen des getesteten Rezepts eines anderen ist ein völlig anderer Vorschlag als das eigene Verfassen.
- **Arbeiten Sie am Syntaxbaum, nicht am Text.** Ein regulärer Ausdruck kann einen Methodenaufruf nicht von einer Zeichenkette unterscheiden, die denselben Namen enthält, und die falsch-positiven Ergebnisse in einer großen Codebasis werden mehr Zeit verbrauchen, als die Migration einspart. Dies ist der gesamte Grund, warum diese Werkzeuge existieren.
- **Führen Sie es zuerst an einem Modul aus** und überprüfen Sie den Diff von Hand, Zeile für Zeile. Die erste Anwendung ist der Ort, an dem Sie entdecken, dass das Rezept ein von Ihrer Codebasis genutztes Muster nicht handhabt, und das an 40 Dateien zu entdecken ist sehr anders, als es an 4.000 zu entdecken.
- **Landen Sie die Migration als eigene Änderung**, ohne jegliche Verhaltensänderung. Eine mechanische Änderung von 4.000 Zeilen ist überprüfbar, wenn der Reviewer weiß, dass sie verhaltensbewahrend ist und stichprobenartig prüfen kann; derselbe Diff mit einer darin versteckten funktionalen Änderung ist überhaupt nicht überprüfbar.
- **Verifizieren Sie mit der Test-Suite, die Sie haben, und seien Sie ehrlich darüber, was sie nicht abdeckt.** Wo die Abdeckung dünn ist, sind Charakterisierungstests um den betroffenen Bereich die Voraussetzung, und dies zu überspringen ist, wie eine mechanische Migration das Verhalten stillschweigend ändert.
- **Schreiben Sie Ihr eigenes Rezept, wenn die Änderung oft genug wiederholt wird, um es zu rechtfertigen** — eine intern abgekündigte API, eine zu standardisierende Logging-Konvention, ein zu ersetzendes veraltetes Utility. Die Schwelle liegt niedriger, als Menschen annehmen, ungefähr ein paar Hundert Aufrufstellen.
- **Halten Sie Rezepte in der Versionskontrolle zusammen mit dem Code** und führen Sie sie periodisch erneut aus. Ein Rezept, das eine Konvention durchsetzt, wird zu einer Möglichkeit, das Wiederauftreten des Musters zu verhindern, nicht nur, es einmal zu entfernen.
- **Behandeln Sie den Rest explizit.** Kein Rezept erreicht 100 Prozent; es wird Aufrufstellen geben, die Reflection, dynamischen Dispatch oder ein Idiom nutzen, das das Rezept nicht erkennt. Listen Sie sie auf, beheben Sie sie von Hand, und lassen Sie den unfertigen Rest nicht das Landen der übrigen neunzig Prozent blockieren.
- **Kombinieren Sie mit einem Qualitäts-Ratschen**, sodass das alte Muster nicht zurückkehren kann: Sobald die Migration landet, hält eine Regel, dass die abgekündigte API nicht wieder eingeführt werden darf, die Codebasis davon ab, zurückzudriften.

## Tradeoffs ⇄

> Rezeptbasierte Migration macht Änderungen möglich, die sonst einfach nicht versucht werden, aber das Tooling hat echte Lernkosten, und mechanische Änderungen in großem Maßstab tragen eigene Risiken.

**Vorteile:**

- Migrationen, die von Hand unpraktikabel sind, werden zur Routine, was direkt den Grund angeht, warum Legacy-Codebasen bei ihren Abhängigkeiten viele Versionen zurückliegen.
- Die Änderung ist überall konsistent, sodass die Codebasis nicht mit dem neuen Idiom in den Dateien endet, zu denen jemand gekommen ist, und dem alten im Rest.
- Es ist weit sicherer als Suchen-und-Ersetzen, weil das Werkzeug Typen und Scope versteht und keinen Text abgleicht, der lediglich ähnlich aussieht.
- Rezepte sind wiederholbar und teilbar, sodass dieselbe Migration auf andere Services angewendet und erneut ausgeführt werden kann, um Regression zu verhindern.
- Schätzungen für große mechanische Änderungen werden glaubwürdig, da die Arbeit größtenteils das Rezept ist statt der Aufrufstellen.

**Kosten und Risiken:**

- Das Tooling hat eine echte Lernkurve, und das Verfassen eines nicht-trivialen Rezepts ist eine Fähigkeit, die Zeit zum Erwerben braucht.
- Enorme mechanische Diffs sind schwierig sinnvoll zu überprüfen, und Reviewer neigen dazu, sie auf Vertrauen zu genehmigen — was nur vernünftig ist, wenn die Verhaltensbewahrungsbehauptung anderweitig verifiziert wird.
- Ohne ausreichende Testabdeckung kann eine mechanische Änderung Verhalten unsichtbar ändern, und Legacy-Codebasen sind genau dort, wo die Abdeckung dünn ist.
- Die Unterstützung ist über Sprachen und Ökosysteme hinweg ungleichmäßig; manche Legacy-Stacks haben überhaupt kein nutzbares Tooling dieser Art.
- Große Migrationen erzeugen Merge-Konflikte mit allem, was in Bearbeitung ist, sodass sie mit der übrigen Arbeit des Teams koordiniert werden müssen statt opportunistisch gelandet zu werden.

## How It Could Be

Ein Team, das eine Java-Plattform pflegte, lag drei Hauptversionen hinter seinem Framework zurück und hatte das Upgrade auf vier Monate geschätzt, fast vollständig für die Anpassung von etwa 5.800 Aufrufstellen an geänderte APIs. Die Schätzung war zweimal abgelehnt worden. Sie führten ein veröffentlichtes Migrationsrezept gegen eines ihrer elf Module aus: 94 Prozent der Aufrufstellen wurden automatisch in unter einer Minute transformiert, und die verbleibenden 6 Prozent wurden aufgelistet. Die Überprüfung des Diffs dieses Moduls von Hand dauerte einen Tag und fand zwei Muster, die das Rezept auf eine Weise handhabte, die sie nicht wollten, welche sie überschrieben. Die Anwendung über die verbleibenden zehn Module dauerte eine Woche, der manuelle Rest dauerte eine weitere Woche, und das gesamte Upgrade landete in unter einem Monat gegen eine Vier-Monats-Schätzung. Die entscheidende Änderung war nicht die Geschwindigkeit des Werkzeugs, sondern dass die Arbeit überprüfbar geworden war — der Framework-Diff war mechanisch und separat, und die elf Verhaltensanpassungen waren eine kleine Änderung, die tatsächlich gelesen werden konnte.

Das Team schrieb anschließend zwei eigene Rezepte. Das erste ersetzte ein internes Datums-Utility, dessen Parameterreihenfolge wiederkehrende Fehler verursacht hatte, über etwa 900 Aufrufstellen. Das zweite setzte ihre Logging-Konvention durch — Korrelationskennung vorhanden, keine String-Konkatenation in Log-Aufrufen — und wurde in der Pipeline ausgeführt statt einmalig, sodass das Muster nicht wieder auftauchen konnte. Das zweite erwies sich als das wertvollere: Ihr vorheriger Versuch, diese Konvention zu etablieren, war eine schriftliche Richtlinie gewesen, der etwa zwei Monate gefolgt wurde, wonach die Einhaltung auf ungefähr den Ausgangspunkt zurückverfiel.
