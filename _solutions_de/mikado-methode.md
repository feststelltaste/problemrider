---
title: Mikado-Methode
description: Entdeckung des tatsächlichen Abhängigkeitsgraphen einer großen
  Änderung durch Ausprobieren, Aufzeichnen, was bricht, Zurücksetzen und
  Beheben der Voraussetzungen zuerst.
category:
- Code
- Process
- Architecture
problems:
- maintenance-paralysis
- large-estimates-for-small-changes
- fear-of-change
- fear-of-breaking-changes
- second-system-effect
- increasing-brittleness
- strangler-fig-pattern-failures
- incomplete-projects
- monolithic-functions-and-classes
- history-of-failed-changes
- analysis-paralysis
- high-coupling-low-cohesion
- past-negative-experiences
- procrastination-on-complex-tasks
- long-lived-feature-branches
- refactoring-avoidance
- complex-implementation-paths
- large-feature-scope
layout: solution
lang: de
en_slug: mikado-method
related_solutions:
- slug: change-impact-analysis
  similarity: 0.75
- slug: small-change-batches
  similarity: 0.75
- slug: large-scale-refactoring
  similarity: 0.75
- slug: preparatory-refactoring
  similarity: 0.75
- slug: incremental-refactoring
  similarity: 0.7
- slug: dependency-breaking-techniques
  similarity: 0.7
---

## Description

Die Mikado-Methode ist eine Technik, um große strukturelle Änderungen an einer Codebasis vorzunehmen, ohne sie je defekt zu hinterlassen. Statt die Änderung im Voraus zu planen — was Kenntnis des Abhängigkeitsgraphen erfordert, die niemand hat —, versuchen Sie die Änderung naiv, beobachten präzise, was bricht, setzen sofort zurück und erfassen die Brüche als Voraussetzungen. Jede Voraussetzung wird dann auf dieselbe Weise versucht, was einen Baum von Abhängigkeiten produziert, empirisch entdeckt statt erraten. Die Arbeit schreitet von den Blättern nach innen voran: Jedes Blatt ist eine Änderung, klein genug, um sie an einem funktionierenden System abzuschließen, zu verifizieren und zu committen. Die zentrale Einsicht der Methode ist, dass der Abhängigkeitsgraph in einer Legacy-Codebasis im Voraus nicht wissbar ist, sodass Compiler und Testsuite als Instrumente genutzt werden sollten, um ihn zu entdecken. Der charakteristische Fehlermodus, den sie verhindert, ist der mehrwöchige Branch, der mit jedem Behebungsversuch defekter wird.

## How to Apply ◆

> Diese Methode existiert speziell für die Situation, in der eine Änderung einfach erscheint, sich herausstellt, ein Dutzend Stellen zu berühren, und der Entwickler drei Tage rein ist, nichts funktioniert und kein Weg zurück besteht.

- Schreiben Sie das **Ziel oben auf ein Blatt oder eine Datei**, formuliert als konkreter Endzustand: „Der ReportGenerator liest das globale Konfigurations-Singleton nicht mehr." Vage Ziele produzieren Bäume, die nie enden.
- **Versuchen Sie das Ziel direkt und naiv.** Nehmen Sie die Änderung vor, als hinge nichts anderes davon ab. Versuchen Sie nicht, etwas zu reparieren, das Sie brechen. Der Zweck dieses Schritts ist Messung, nicht Fortschritt.
- **Erfassen Sie jeden Fehler**, den Compiler, Build oder Testsuite produzieren, als Kandidat-Voraussetzung. Seien Sie spezifisch: benennen Sie die Datei und den Grund, nicht „Tests schlagen fehl". Die Qualität des Baums hängt vollständig von der Präzision dieser Erfassung ab.
- **Setzen Sie sofort und vollständig zurück.** Dies ist die Disziplin, die die Methode funktionieren lässt, und der Schritt, den Menschen überspringen. Der Arbeitsbaum kehrt nach jedem Experiment zu einem bekannt-guten Zustand zurück, sodass es nie ein halb migriertes System und nie einen Grund gibt zu befürchten, die Änderung könnte nicht aufgegeben werden.
- Versuchen Sie jede **Voraussetzung auf dieselbe Weise**, rekursiv. Voraussetzungen, die abgeschlossen werden können, ohne etwas zu brechen, sind Blätter; solche, die Dinge brechen, erzeugen ihre eigenen Voraussetzungen. Der Baum stellt sich meist als tiefer und schmaler heraus als erwartet, was selbst wertvolle Information ist.
- **Schließen Sie die Blätter ab und committen Sie sie** eines nach dem anderen, jedes an einem funktionierenden System mit bestehenden Tests. Jedes Blatt ist unabhängig wertvoll und unabhängig zurücksetzbar, sodass die Arbeit an jedem Punkt pausiert werden kann, ohne Trümmer zu hinterlassen.
- **Versuchen Sie das ursprüngliche Ziel periodisch erneut**, während Blätter abgeschlossen werden. Es wird oft früher erreichbar, als der Baum nahelegt, weil sich mehrere Voraussetzungen als von derselben zugrundeliegenden Ursache geteilt herausstellen.
- Halten Sie den Baum **für das Team sichtbar** — eine Datei im Repository, ein gemeinsames Diagramm, eine Reihe von Tickets. Er kommuniziert Fortschritt an Arbeit, die sonst so aussieht, als geschähe nichts, und erlaubt jemand anderem, die Anstrengung fortzusetzen, was für Änderungen zählt, die sich über Wochen erstrecken.
- Nutzen Sie **Größe und Form des Baums als Entscheidungsinput**. Ein Baum, der nach zwei Runden auf sechzig Knoten anwächst, sagt Ihnen etwas über die echten Kosten der Änderung, früh genug, um Umfang oder Ansatz zu überdenken, statt es in Woche fünf zu entdecken.

## Tradeoffs ⇄

> Die Methode tauscht scheinbare Effizienz gegen echte Sicherheit: wiederholtes Zurücksetzen fühlt sich verschwenderisch an, und ist der Grund, warum die Codebasis nie einen Tag in einem defekten Zustand verbringt.

**Vorteile:**

- Das System ist an jedem Punkt funktionsfähig und committbar, sodass die Änderung pausiert, übergeben oder aufgegeben werden kann, ohne Verlust — was den größten Teil des Risikos beseitigt, das große Legacy-Änderungen furchteinflößend macht.
- Der echte Abhängigkeitsgraph wird entdeckt statt geschätzt, weshalb die Methode nützliche Kosteninformation für Änderungen produziert, die Vorabanalyse systematisch unterschätzt.
- Arbeit wird natürlich in kleine, unabhängig überprüfbare Commits zerlegt, ohne dass der Entwickler diese Zerlegung im Voraus entwerfen muss.
- Analyse-Paralyse wird kurzgeschlossen: Der Weg herauszufinden, was eine Änderung erfordert, ist, sie zwanzig Minuten zu versuchen, nicht den Code zwei Tage zu studieren.
- Teilweise abgeschlossene Bemühungen hinterlassen die Codebasis besser statt schlechter, da jedes committete Blatt eine echte Verbesserung ist, selbst wenn das Ziel nie erreicht wird.

**Kosten und Risiken:**

- Der wiederholte Zurücksetzen-und-erneut-versuchen-Zyklus ist pro Versuch echt langsamer und fühlt sich unproduktiv an, besonders für Entwickler unter Zeitdruck und für Beobachter, die Commit-Aktivität verfolgen.
- Sie hängt von schnellem Feedback ab. Wenn der Build-und-Test-Zyklus vierzig Minuten dauert, ist die Schleife zu langsam, um praktikabel zu sein, und die Build-Zeit muss zuerst adressiert werden.
- Ohne automatisierte Tests kann Bruch nicht zuverlässig erkannt werden, sodass die Methode auf reine Compiler-getriebene Entdeckung degradiert — nützlich in statisch typisierten Sprachen, schwach in dynamisch typisierten.
- Der Baum kann groß genug werden, um demoralisierend zu wirken, und Teams geben die Bemühung manchmal an diesem Punkt auf, statt den Baum als die genaue Kostenschätzung zu lesen, die er ist.
- Disziplin ist beim Zurücksetzen nötig. Ein einziges „ich repariere das nur schnell, während ich hier bin" führt den defekten-Branch-Fehlermodus wieder ein, den die Methode verhindern soll.

## How It Could Be

Eine Entwicklerin, die einen Versandpreisrechner pflegte, wurde gebeten, ihn testbar zu machen, damit eine Preisänderung vor dem Release verifiziert werden konnte. Die Klasse las direkt aus einem statischen Konfigurationshalter, einem Datenbank-Singleton und der Systemuhr. Ihre ersten drei Versuche über zwei Wochen hatten jeweils mit einem zu defekten Branch geendet, um ihn fertigzustellen, und waren aufgegeben worden. Mit der Mikado-Methode brach ihr erster naiver Versuch — die Entfernung der statischen Konfigurationsreferenz — innerhalb von neun Minuten elf Kompilationseinheiten, und sie setzte zurück. Der entstandene Baum hatte vier Voraussetzungen, von denen eine drei eigene hatte. Über neun Tage committete sie vierzehn kleine Änderungen, jede mit grünem Build, und die vierzehnte ließ das ursprüngliche Ziel beim ersten Versuch gelingen. Die Preisänderung, die die Arbeit ausgelöst hatte, brauchte einen Nachmittag.

Ein Team, das versuchte, ein Kundenmodul aus einem Monolithen zu extrahieren, nutzte die Methode, um sich gegen die Extraktion im geplanten Umfang zu entscheiden. Zwei Runden naiver Versuche produzierten einen Baum mit über fünfzig Knoten, dominiert von einem gemeinsamen Datenbankschema, das elf andere Module direkt lasen. Statt fortzufahren, nutzten sie den Baum als Beleg in einer Planungsdiskussion: Die Extraktion war keine Vier-Wochen-Aufgabe, sondern eine über mehrere Quartale, und ihr echter erster Schritt war Schema-Eigentümerschaft, nicht Code-Verschiebung. Sie committeten die acht Blätter, die sie bereits als unabhängig wertvoll identifiziert hatten — meist Schnittstellenextraktionen, die die Testbarkeit verbesserten — und lenkten die verbleibende Anstrengung auf das Schema um. Der Baum wurde zum Planungsdokument für die folgenden zwei Quartale.
