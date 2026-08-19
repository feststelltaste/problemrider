---
title: Vorbereitendes Refactoring
description: Vor einer Änderung den Code so umstrukturieren, dass die
  Änderung einfach wird — dann die einfache Änderung vornehmen, als zwei
  getrennte Schritte.
category:
- Code
- Process
problems:
- refactoring-avoidance
- large-estimates-for-small-changes
- feature-creep-without-refactoring
- complex-implementation-paths
- defensive-coding-practices
- accumulation-of-workarounds
- increasing-brittleness
- copy-paste-programming
- maintenance-paralysis
- increased-technical-shortcuts
- difficult-to-understand-code
- workaround-culture
- bloated-class
- global-state-and-side-effects
- god-object-anti-pattern
- long-lived-feature-branches
- merge-conflicts
- convenience-driven-development
- monolithic-functions-and-classes
- over-reliance-on-utility-classes
- poor-encapsulation
- procrastination-on-complex-tasks
- tangled-cross-cutting-concerns
layout: solution
lang: de
en_slug: preparatory-refactoring
related_solutions:
- slug: incremental-refactoring
  similarity: 0.8
- slug: small-change-batches
  similarity: 0.75
- slug: mikado-method
  similarity: 0.75
- slug: large-scale-refactoring
  similarity: 0.7
- slug: code-review-process-reform
  similarity: 0.7
- slug: refactoring-katas
  similarity: 0.7
---

## Description

Vorbereitendes Refactoring bedeutet, dass Sie, wenn eine Änderung schwer umzusetzen ist, zuerst den Code so umstrukturieren, dass die Änderung einfach wird, und sie erst dann vornehmen — als zwei getrennte, separat verifizierte Schritte. Es löst das wiederkehrende Dilemma der Legacy-Wartung: Der Code ist nicht für die angeforderte Änderung geformt, also erzwingt der Entwickler entweder die Änderung in die bestehende Form, wodurch ein weiterer Workaround entsteht, oder er begibt sich auf eine offene Bereinigung, die schwer zu rechtfertigen und schwer zu überprüfen ist. Die Disziplin vermeidet beides. Das Refactoring ist durch einen spezifischen Zweck begrenzt — diese bestimmte Änderung einfach machen —, was es beschränkt, und es ist verhaltensbewahrend, was es durch bestehende Tests verifizierbar macht. Die nachfolgende funktionale Änderung ist dann klein genug, um ordentlich überprüft zu werden. Die Praxis verwandelt Verbesserung außerdem von einer separaten Aktivität, die finanziert werden muss, in einen normalen Teil der Arbeit.

## How to Apply ◆

> Legacy-Code ist durch die Änderungen geformt, die an ihm vorgenommen wurden, weshalb die nächste Änderung nie passt: Die Form spiegelt Anforderungen von vor Jahren wider, die niemand seither überprüft hat.

- **Versuchen Sie die Änderung zuerst, kurz.** Die Schwierigkeit sagt Ihnen, welche Umstrukturierung benötigt wird. Spekulativ zu refaktorieren, bevor bekannt ist, was die Änderung erfordert, produziert eine andere Form, in die die Änderung immer noch nicht passt.
- **Machen Sie diesen Versuch rückgängig** und führen Sie das Refactoring eigenständig durch. Beides zu vermischen ist der Fehlerfall: Ein Diff, der sowohl eine Verschiebung als auch eine Verhaltensänderung enthält, kann nicht überprüft werden, weil der Reviewer nicht erkennen kann, welche Zeilen verhaltensneutral sein sollen.
- Halten Sie das Refactoring **auf das beschränkt, was die Änderung braucht.** „Diese Methode extrahieren, damit der neue Zweig einen Platz hat" ist begrenzt und verteidigbar; „diese Klasse aufräumen" ist beides nicht, und das ist es, was Manager Refactoring misstrauen lässt.
- **Committen und idealerweise das Refactoring separat ausliefern.** Es ist verhaltensbewahrend, trägt also geringes Risiko und kann eigenständig in Produktion gehen. Wenn die funktionale Änderung dann verzögert oder abgesagt wird, hat sich die Codebasis trotzdem verbessert.
- **Verifizieren Sie Verhaltensbewahrung** mit welchem Sicherheitsnetz auch immer existiert — den bestehenden Tests oder eigens dafür geschriebenen Characterization Tests. Wo keines existiert, verwenden Sie konservative, abhängigkeitsbrechende Transformationen, klein genug, um durch Lesen zu verifizieren.
- Beziehen Sie den vorbereitenden Schritt **in die Schätzung ein, statt ihn zu verstecken.** Ihn als Teil der Arbeit zu präsentieren ist ehrlich und macht die echten Kosten der aktuellen Form des Codes sichtbar. Ihn als Puffer zu verbergen untergräbt Vertrauen, wenn es entdeckt wird.
- Erkennen Sie das **Signal, dass der Code Ihnen etwas sagt**. Eine Änderung, die das Anfassen von sechs Stellen erfordert, sagt Ihnen, dass diese sechs Stellen ein Konzept teilen, das nie benannt wurde. Das Refactoring, das das behebt, ist das wertvolle.
- Wissen Sie, wann Sie **nicht vorbereiten sollten**: Code, der zur Löschung geplant ist, Code, der sich seit Jahren nicht geändert hat und sich jetzt nicht ändert, und echte Notfälle. Vorbereitendes Refactoring verdient seine Kosten dort, wo Änderung fortlaufend ist.
- Wenn sich herausstellt, dass die Vorbereitung **weit größer ist als die Änderung**, stoppen Sie und behandeln Sie es als eigenes Arbeitsstück mit eigener Entscheidung. Das zu entdecken ist ein nützliches Ergebnis, und trotzdem weiterzumachen ist, wie eine Zwei-Tage-Aufgabe zu einem Drei-Wochen-Branch wird.

## Tradeoffs ⇄

> Die Arbeit aufzuteilen macht beide Hälften überprüfbar und Verbesserung routinemäßig, zum Preis eines längeren Wegs zur funktionalen Änderung und einer Disziplin, die unter Druck erodiert.

**Vorteile:**

- Beide Schritte werden einzeln überprüfbar, was eine erhebliche Qualitätsverbesserung gegenüber einem einzelnen Diff ist, der Verschiebung mit Verhaltensänderung vermischt.
- Die funktionale Änderung endet klein, offensichtlich und sicher, woher der Großteil der Fehlerreduktion kommt.
- Verbesserung geschieht kontinuierlich, in den Bereichen, die tatsächlich geändert werden, ohne separate Finanzierung oder eine dedizierte Initiative zu benötigen.
- Das Refactoring ist risikoarm und unabhängig auslieferbar, sodass teilweise abgeschlossene Arbeit die Codebasis trotzdem besser hinterlässt.
- Es wirkt der Anhäufung von Workarounds direkt entgegen, weil die Alternative — die Änderung in eine nicht passende Form zu zwingen — genau ist, wie sich Workarounds ansammeln.

**Kosten und Risiken:**

- Die funktionale Änderung braucht länger, bis sie ankommt, und unter Terminendruck ist die Vorbereitung der Schritt, der übersprungen wird, gerade wenn er am meisten gebraucht wurde.
- Ohne Tests kann Verhaltensbewahrung nicht verifiziert werden, sodass die Praxis entweder ein Sicherheitsnetz braucht oder sich auf durch Lesen verifizierbare Transformationen beschränken muss.
- Umfangsdisziplin ist echt schwer. Vorbereitendes Refactoring driftet leicht in allgemeine Bereinigung ab, und sobald das passiert, wird es zur unbegrenzten Aktivität, die das Management ablehnen lernt.
- Auf die falsche Änderung ausgerichtete Vorbereitung verschwendet Aufwand und kann den Code für eine Anforderung geformt zurücklassen, die nie eintrifft.
- Zwei Commits, wo einer war, fügt Prozess-Overhead hinzu, was real ist, wenn der Build- und Review-Zyklus langsam ist.

## How It Could Be

Ein Entwickler wurde gebeten, einem Checkout-Ablauf einen zweiten Rabatttyp hinzuzufügen. Die bestehende Rabattlogik war eine 200-Zeilen-Methode mit dem einzigen Rabatttyp, der durch sechs bedingte Verzweigungen gewoben war. Ihr erster Versuch, den neuen Typ direkt hinzuzufügen, produzierte eine Methode, der sie nach zwanzig Minuten nicht mehr folgen konnte. Sie machte es rückgängig, verbrachte einen Tag damit, die Rabattberechnung in eine kleine Schnittstelle mit der bestehenden Logik als einziger Implementierung zu extrahieren, verifizierte gegen die bestehenden Tests und lieferte das als eigene Änderung aus. Am nächsten Tag war der neue Rabatttyp eine neue Implementierung dieser Schnittstelle: 40 Zeilen, mit eigenen Tests und einer Zwei-Zeilen-Änderung an der Aufrufstelle. Die gesamte verstrichene Zeit war etwas mehr, als eine erzwungene Änderung gebraucht hätte. Als vier Monate später ein dritter Rabatttyp angefragt wurde, dauerte es einen Nachmittag.

Ein Team nutzte dieselbe Disziplin, um ein Modernisierungsargument zu machen. Ihre Nachverfolgung zeigte, dass sich ein Drittel ihrer vorbereitenden Refactorings als größer herausstellte als die Änderung, die sie auslöste, und dass diese sich in zwei Subsystemen häuften. Statt weiterhin die Kosten unsichtbar innerhalb einzelner Aufgaben zu absorbieren, zeichneten sie es auf: Über ein Quartal waren 31 Entwicklertage Vorbereitung in diesen zwei Subsystemen für Änderungen verbraucht worden, deren funktionaler Inhalt 6 Tage betrug. Das Verhältnis, präsentiert als gemessene Zahl statt als Beschwerde, war das, was einen dedizierten Aufwand für das schlechtere der beiden sicherte — und es war vollständig aus Arbeit abgeleitet, die das Team ohnehin schon tat.
