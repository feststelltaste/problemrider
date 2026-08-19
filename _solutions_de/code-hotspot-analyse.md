---
title: Code-Hotspot-Analyse
description: Kombination von Änderungshäufigkeit aus der Versionskontrolle mit Komplexitäts-
  und Fehlerdaten, um den kleinen Anteil an Code zu identifizieren, bei dem sich
  Verbesserungsaufwand tatsächlich lohnt.
category:
- Code
- Process
- Management
problems:
- maintenance-bottlenecks
- bloated-class
- excessive-class-size
- copy-paste-programming
- increasing-brittleness
- increased-bug-count
- maintenance-cost-increase
- high-technical-debt
- invisible-nature-of-technical-debt
- monolithic-functions-and-classes
- delayed-issue-resolution
- automated-tooling-ineffectiveness
- feature-creep-without-refactoring
- system-stagnation
- god-object-anti-pattern
- refactoring-avoidance
- code-duplication
- delayed-bug-fixes
- maintenance-paralysis
- tangled-cross-cutting-concerns
- accumulation-of-workarounds
- brittle-codebase
- quality-degradation
layout: solution
lang: de
en_slug: code-hotspot-analysis
related_solutions:
- slug: technical-debt-assessment
  similarity: 0.75
- slug: code-metrics
  similarity: 0.75
- slug: technical-debt-backlog
  similarity: 0.75
- slug: static-analysis-and-linting
  similarity: 0.75
- slug: change-impact-analysis
  similarity: 0.7
- slug: debt-accrual-analysis
  similarity: 0.7
---

## Description

Code-Hotspot-Analyse identifiziert, wo sich Verbesserungsaufwand am meisten auszahlt, indem zwei Datenquellen gekreuzt werden, die die meisten Teams bereits haben: wie oft sich jede Datei ändert, entnommen aus der Versionskontrollhistorie, und wie komplex oder fehleranfällig jede Datei ist. Komplexität allein ist ein schlechter Leitfaden — ein kompliziertes Modul, das niemand seit fünf Jahren geändert hat, kostet nichts, es in Ruhe zu lassen. Änderungshäufigkeit allein ist ebenso schlecht, da eine häufig geänderte, aber einfache Datei kein Problem ist. Ihre Schnittmenge ist klein, typischerweise wenige Prozent der Dateien, und macht konsistent einen unverhältnismäßigen Anteil an Defekten, Review-Zeit und Entwicklungsaufwand aus. In einem Legacy-System, wo alles schlecht aussieht und der verfügbare Aufwand einen Bruchteil dessen ausmacht, was eine vollständige Bereinigung erfordern würde, ist der Wert der Analyse, dass sie die Frage beantwortet, die Teams sonst nicht beantworten können: von all dem, was reparieren wir zuerst?

## How to Apply ◆

> Die Versionskontrollhistorie eines langlebigen Systems ist ein unterschätztes Protokoll davon, wo der Schmerz tatsächlich liegt, und sie ist verfügbar, ohne irgendetwas zu instrumentieren oder jemanden zu fragen.

- **Extrahieren Sie Änderungshäufigkeit pro Datei** aus dem Repository-Log über ein bedeutsames Fenster — üblicherweise ein bis zwei Jahre. Kürzere Fenster sind verrauscht; längere beinhalten Churn von einem System, das nicht mehr existiert. Schließen Sie Massen-Formatierungs- und Umbenennungs-Commits aus, die sonst die Zählungen dominieren und irreführende Ergebnisse produzieren.
- **Paaren Sie Häufigkeit mit einem Komplexitäts-Proxy**: Codezeilen ist grob, funktioniert aber überraschend gut; zyklomatische Komplexität oder Einrückungstiefe sind besser, wo ein Werkzeug verfügbar ist. Plotten Sie die beiden Achsen und schauen Sie sich den oberen rechten Quadranten an. Dieser Quadrant ist die Hotspot-Menge, und sie ist üblicherweise erstaunlich klein.
- **Überlagern Sie Fehlerdaten**, indem Sie Commits mit Bug-Tickets abgleichen, wenn Commit-Nachrichten oder Branch-Namen dies erlauben. Dateien, die häufig geändert, komplex und wiederholt in Defekte verwickelt sind, sind die Ziele mit dem höchsten Vertrauen in der Codebasis.
- Analysieren Sie **zeitliche Kopplung** — Dateien, die wiederholt im selben Commit geändert werden, trotz keiner expliziten Abhängigkeit. Diese Paare offenbaren versteckte Kopplung, die keine statische Analyse findet, und sie sind oft der klarste Beweis für eine fehlende Abstraktion oder eine undichte Grenze.
- Schauen Sie sich die **Autorenverteilung pro Hotspot** an. Ein Hotspot mit einem Mitwirkenden ist sowohl ein Wissens- als auch ein Coderisiko; ein Hotspot mit zwanzig Mitwirkenden und keinem Eigentümer hat üblicherweise Konsistenzprobleme. Die beiden Situationen verlangen unterschiedliche Antworten.
- **Führen Sie die Analyse nach Plan erneut aus** — vierteljährlich ist typisch — und verfolgen Sie, ob die Hotspot-Menge schrumpft. Ein Hotspot, der adressiert wurde, sollte innerhalb weniger Zyklen aus dem oberen rechten Quadranten fallen; wenn nicht, hat die Intervention nicht funktioniert, und das ist wissenswert.
- Nutzen Sie die Ausgabe, um das **Verbesserungsbudget zu lenken**, statt Teams oder Einzelpersonen zu bewerten. In dem Moment, in dem Hotspot-Daten in Leistungsbeurteilungen genutzt werden, ändert sich das Commit-Verhalten, um die Metrik zu optimieren, und die Daten hören auf, das System zu beschreiben.
- Präsentieren Sie die Analyse **visuell für nicht-technische Stakeholder**. Eine Treemap, in der die Größe die Änderungshäufigkeit und die Farbe die Komplexität ist, kommuniziert den Fall für Wartungsinvestition weit effektiver als jedes verbale Argument über technische Schulden, weil sie ein unsichtbares Problem sichtbar macht.
- **Validieren Sie Hotspots gegen die Erfahrung des Teams**, bevor Sie handeln. Die Analyse identifiziert Kandidaten; Entwickler wissen, welche davon genuin schmerzhaft und welche nur groß sind. Wo Daten und Team nicht übereinstimmen, ist die Meinungsverschiedenheit üblicherweise aufschlussreich.

## Tradeoffs ⇄

> Hotspot-Analyse ist günstig, evidenzbasiert und lenkt Aufwand gut, aber sie misst Proxys statt Qualität und kann aktiv irreführend sein, wenn naiv gelesen.

**Vorteile:**

- Verbesserungsaufwand wird auf den kleinen Teil der Codebasis gelenkt, wo er etwas ändert, statt gleichmäßig über Code verteilt zu werden, der größtenteils inaktiv ist.
- Technische Schulden werden sichtbar und quantifiziert, was generell die fehlende Zutat in Gesprächen mit Stakeholdern über die Finanzierung von Wartung ist.
- Die Analyse kostet sehr wenig — ein Skript über die bestehende Repository-Historie — und erfordert keine Kooperation von irgendjemandem.
- Zeitliche Kopplung offenbart architektonische Probleme, die keine statische Analyse erkennt, oft genau identifizierend, wo eine Grenze fehlt.
- Fortschritt kann über die Zeit gemessen werden, sodass Verbesserungsarbeit eine Metrik erhält, die nicht nur die Behauptung des Teams ist, dass Dinge besser sind.

**Kosten und Risiken:**

- Änderungshäufigkeit misst Aktivität, nicht Qualität. Eine Datei unter aktiver Feature-Entwicklung ist nicht notwendigerweise ein Problem, und sie als solches zu behandeln verschwendet Aufwand und irritiert das Team, das daran baut.
- Repository-Historie verzerrt sich leicht: Datei-Umbenennungen, Verschiebungen, Massenumformatierung und Repository-Migrationen korrumpieren alle die Zählungen, es sei denn, sie werden explizit gehandhabt.
- Codezeilen und zyklomatische Komplexität sind schwache Proxys für das, was tatsächlich zählt, nämlich wie schwierig der Code korrekt zu ändern ist.
- Die Analyse wird kein Modul zutage bringen, das gefährlich, aber selten berührt ist — einschließlich Code, der genau deshalb gemieden wird, weil er beängstigend ist, was ein echter blinder Fleck in Legacy-Kontexten ist.
- Als Leistungsmaß genutzt, korrumpieren sich die Daten sofort, da Commit-Granularität und Dateiorganisation trivial manipulierbar sind.

## How It Could Be

Ein Team, das ein 900.000-Zeilen-ERP-System pflegte, hatte einen technischen-Schulden-Backlog mit über 200 Einträgen und keine Möglichkeit, ihn zu ordnen. Sie führten eine Hotspot-Analyse über 18 Monate Historie durch und fanden, dass 14 Dateien — unter einem Prozent der Codebasis — 31 Prozent aller Commits ausmachten und in 44 Prozent der mit Bug-Tickets verknüpften Commits erschienen. Vier dieser Dateien waren überhaupt nicht im Schulden-Backlog, weil sie unangenehm statt offensichtlich kaputt waren und niemand sie vorgeschlagen hatte. Das Team leitete sein Verbesserungsbudget für zwei Quartale zu den sechs Top-Hotspots um. Fehlerberichte, die diesen Dateien zuzuschreiben waren, sanken um ungefähr zwei Drittel, und der nächste Hotspot-Lauf zeigte, dass alle sechs aus dem oberen rechten Quadranten gefallen waren.

Die zeitliche-Kopplungs-Ausgabe derselben Analyse produzierte einen folgenreicheren Befund. Zwei Dateien in nominell separaten Subsystemen — ein Auftragsmodul und ein Lagermodul — änderten sich in 78 Prozent der Commits zusammen, die eines von beiden berührten, trotz keiner Import-Beziehung zwischen ihnen. Untersuchung fand eine undokumentierte gemeinsame Annahme über eine Statuscode-Enumeration, dupliziert an beiden Orten. Sie hatte drei Produktionsvorfälle über zwei Jahre verursacht, jeder separat untersucht, und keiner hatte das Muster identifiziert. Das Team vereinheitlichte die Enumeration in einer Woche. Die visuelle Treemap aus dieser Analyse war außerdem, was der Engineering-Manager nutzte, um das Wartungsbudget des folgenden Jahres zu sichern, nachdem zwei vorherige Versuche mit verbalen Argumenten gescheitert waren.
