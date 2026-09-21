---
title: Werthierarchie
description: Pflege einer expliziten Kette von jedem technischen
  Arbeitsschritt bis zu einem Geschäftsziel, sodass Wert nachverfolgt
  statt behauptet werden kann.
category:
- Business
- Management
- Architecture
problems:
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- unclear-goals-and-priorities
- feature-factory
- short-term-focus
- product-direction-chaos
- invisible-nature-of-technical-debt
- competing-priorities
- wasted-development-effort
- delayed-value-delivery
- system-stagnation
- reduced-innovation
- competitive-disadvantage
- declining-business-metrics
- feature-bloat
- feedback-isolation
- high-maintenance-costs
- increased-cost-of-development
- market-pressure
- project-resource-constraints
- resource-waste
- slow-development-velocity
- stakeholder-confidence-loss
- stakeholder-frustration
- high-technical-debt
layout: solution
lang: de
en_slug: value-hierarchy
related_solutions:
- slug: impact-mapping
  similarity: 0.65
- slug: product-strategy-alignment
  similarity: 0.65
- slug: explicit-prioritization-framework
  similarity: 0.65
- slug: technical-debt-backlog
  similarity: 0.65
- slug: decision-rights-and-escalation
  similarity: 0.65
- slug: mikado-method
  similarity: 0.65
---

## Description

Eine Werthierarchie ist eine explizite, gepflegte Argumentationskette, die jedes Stück technischer Arbeit über Zwischenschritte mit einem Geschäftsziel verbindet: diese Refaktorierung verkürzt den Änderungszyklus in diesem Subsystem, was die Markteinführungszeit für diese Produktlinie verkürzt, was dem diesjährigen erklärten Ziel dient, schneller als ein benannter Wettbewerber zu reagieren. Sie existiert, weil das Argument für technische Arbeit fast immer eine Kette mehrerer Glieder ist, und technische Personen gewohnheitsmäßig nur das erste Glied nennen, während Geschäftspersonen nur das letzte bewerten können. Die Lücke zwischen ihnen ist, wo Modernisierungsvorschläge sterben — nicht weil die Verbindung fehlt, sondern weil niemand sie niedergeschrieben hat und jede Seite annimmt, die andere solle sie sehen. Die Kette explizit zu machen tut zwei Dinge: Es lässt Wert Glied für Glied nachverfolgt und hinterfragt werden, und es bringt die Arbeit zutage, deren Kette tatsächlich nirgendwohin führt.

*Die Idee, Wert in eine explizite Hierarchie zu ordnen, stammt aus der Cloud-Native-Patterns-Community, wo sie als Strategiemuster für Transformationsanstrengungen erscheint.*

## How to Apply ◆

> In einem Legacy-Kontext ist die Kette üblicherweise drei oder vier Glieder lang, und die mittleren Glieder — die über Änderungskosten und Risiko — sind genau die, die niemand außerhalb der Entwicklung liefern kann.

- **Beginnen Sie mit den bereits bestehenden Geschäftszielen**, in den Worten, die die Organisation bereits nutzt. Eine Hierarchie, gebaut auf von der Entwicklung erfundenen Zielen, ist ein Entwicklungsdokument, und es wird als solches gelesen werden.
- **Schreiben Sie jedes Glied als eine Behauptung, die falsch sein könnte.** "Die Reduzierung des Builds von 30 Minuten auf 5 erhöht die Anzahl der Änderungen, die wir pro Monat liefern können" ist überprüfbar. "Verbessert die Entwicklererfahrung" ist es nicht, und ein nicht falsifizierbares Glied bricht die Kette, wo auch immer es erscheint.
- **Bestehen Sie darauf, dass jede Kette endet.** Arbeit, deren Kette nach zwei Gliedern ausläuft — sie macht den Code schöner, und dann nichts — sollte entweder umformuliert werden, bis sie etwas erreicht, oder als Arbeit anerkannt werden, die um ihrer selbst willen getan wird. Beide Ergebnisse sind nützlicher als eine Kette, die verläuft.
- **Halten Sie die Zwischenschicht ehrlich.** Die mittleren Glieder betreffen üblicherweise Änderungskosten, Risiko und Kapazität, und dies ist, wo die Entwicklung Wissen hat, das niemand sonst hat. Diese Schicht ist der tatsächliche Beitrag des Teams zum Argument; eine Hierarchie, die direkt von einer Refaktorierung zum Umsatz springt, wird nicht geglaubt werden.
- **Fügen Sie Messgrößen an, wo sie existieren**, an welchem Glied auch immer sie existieren. Nicht jedes Glied kann gemessen werden, aber eine Kette mit zwei gemessenen Gliedern ist weit stärker als eine ohne, und die gemessenen Glieder verankern die ungemessenen.
- **Nutzen Sie sie in beide Richtungen.** Nach unten verwandelt sie ein Ziel in einen Satz Kandidaten für technische Investition. Nach oben verwandelt sie ein vorgeschlagenes Arbeitsstück in eine Rechtfertigung. Die Aufwärtsrichtung ist es, was Teams brauchen; die Abwärtsrichtung ist es, was die Führung überhaupt dazu bringt, die Hierarchie zu nutzen.
- **Überprüfen Sie sie, wenn sich Ziele ändern.** Eine Hierarchie, gebaut auf den Zielen des letzten Jahres, rechtfertigt still Arbeit, die nichts mehr dient, und dies ist ein häufiger Weg, wie Modernisierungsprogramme ihre Begründung überleben.
- **Lassen Sie sie Vorschläge töten.** Eine Hierarchie, die nie dazu geführt hat, dass Arbeit fallengelassen wurde, ist dekorativ. Der Wert entsteht daraus, dass sie auf Arbeit angewendet wird, die das Team tun will, nicht nur auf Arbeit, die es verteidigt.
- **Halten Sie sie klein.** Ein Diagramm mit 200 Knoten wird nicht gepflegt oder gelesen. Eine Handvoll Ziele, jedes mit ein paar Ketten darunter, ist es, was nutzbar bleibt.

## Tradeoffs ⇄

> Eine explizite Kette macht technischen Wert argumentierbar statt behauptbar, aber sie erfordert Pflege und kann genutzt werden, um Arbeit abzulehnen, deren Wert echt und schwer zu artikulieren ist.

**Vorteile:**

- Technische Arbeit erwirbt eine erklärte Verbindung zu Geschäftsergebnissen, was es ihr erlaubt, um Finanzierung zu konkurrieren, statt als Overhead klassifiziert zu werden.
- Schwache Glieder werden sichtbar und können spezifisch diskutiert werden, was weit produktiver ist als eine allgemeine Meinungsverschiedenheit darüber, ob technische Arbeit zählt.
- Arbeit, deren Kette nirgendwohin führt, wird identifiziert, und ein Teil davon stellt sich als genuin optional heraus.
- Das besondere Wissen der Entwicklung — über Änderungskosten und Risiko — besetzt einen definierten Platz im Argument, statt das gesamte Argument zu sein.
- Dieselbe Struktur dient der Priorisierung, da Ketten, die die höchstgewichteten Ziele mit dem geringsten Aufwand erreichen, die offensichtlichen Kandidaten sind.

**Kosten und Risiken:**

- Die Pflege der Hierarchie ist laufende Arbeit, und sie veraltet schnell, wenn sich Ziele verschieben, an welchem Punkt sie die falschen Dinge rechtfertigt.
- Ketten können konstruiert werden, um jede Schlussfolgerung zu erreichen. Ein entschlossener Befürworter kann fast jede Arbeit mit fast jedem Ziel durch genug plausibel klingende Glieder verbinden.
- Arbeit mit echtem, aber schwer artikulierbarem Wert — die Reduzierung eines Risikos, das noch niemand erlebt hat, das Offenhalten einer Option — wird systematisch benachteiligt durch ein Framework, das eine explizite Kette verlangt.
- Die Übung kann zu einem Compliance-Ritual werden, das Vorschlägen nach der Entscheidung angehängt wird, was Dokumentation produziert, ohne irgendetwas zu ändern.
- Lange Ketten sind schwache Ketten: Jedes Glied vervielfacht die Unsicherheit, und ein Vier-Glieder-Argument kann aufgrund von Zweifeln an jedem einzelnen abgelehnt werden.

## How It Could Be

Die Vorschläge eines Plattformteams wurden konsistent abgelehnt, während Produkt-Feature-Anfragen genehmigt wurden, und beide Seiten hatten geschlossen, dass die andere das Geschäft nicht verstand. Das Team baute eine Hierarchie, ausgehend von den drei Zielen im veröffentlichten Jahresplan des Unternehmens. Unter "Zeit von Kundenanfrage bis gelieferter Änderung reduzieren" platzierten sie die Zwischenbehauptung, dass die mittlere Änderung am Bestellsubsystem 19 Tage dauerte, von denen 11 auf eine gemeinsam genutzte Testumgebung warteten, und darunter die spezifische Arbeit: ephemere Per-Branch-Umgebungen. Drei Glieder, zwei davon gemessen. Der Vorschlag wurde in einem Meeting genehmigt, nach zwei Jahren Ablehnungen. Das entscheidende Element war nicht die Umgebungen, die zuvor vorgeschlagen worden waren, sondern das mittlere Glied — die 11 Tage —, das niemand außerhalb des Teams gewusst hatte und das kein vorheriger Vorschlag angegeben hatte.

Die Hierarchie beendete auch ein Projekt, das das Team wollte. Sie hatten seit achtzehn Monaten für eine Migration von einem Message-Broker zu einem anderen plädiert. Als sie die Kette ehrlich bauten, fanden sie, dass sie bei "der neuere Broker ist besser unterstützt und wir würden lieber damit arbeiten" ankam und dort stoppte — keine Verbindung zu Änderungskosten, keine Verbindung zu Risiko, das der bestehende Broker tatsächlich produzierte, und kein Ziel, dem er diente. Das Team ließ es fallen. Zwei Entwickler waren darüber unglücklich, und dieselbe Disziplin, ein Quartal später angewendet, produzierte eine Kette für einen Datenbankverbindungspool-Fix, die sauber durch Vorfallstunden bis zu einer Verfügbarkeitsverpflichtung in einem Kundenvertrag lief. Diese Arbeit war nie vorgeschlagen worden, weil niemand daran gedacht hatte, sie als interessant zu betrachten.
