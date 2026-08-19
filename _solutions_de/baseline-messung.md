---
title: Baseline-Messung
description: Messung des aktuellen Zustands, bevor man ihn ändert, denn ein Nutzen
  ohne "Vorher" lässt sich hinterher nie nachweisen.
category:
- Process
- Management
- Business
problems:
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- invisible-nature-of-technical-debt
- planning-credibility-issues
- high-maintenance-costs
- maintenance-cost-increase
- short-term-focus
- quality-degradation
- increasing-brittleness
- slow-development-velocity
- resource-waste
- wasted-development-effort
- budget-overruns
- declining-business-metrics
- deployment-risk
- increased-cost-of-development
- legacy-system-documentation-archaeology
- poor-planning
- reduced-predictability
- regulatory-compliance-drift
- stakeholder-confidence-loss
- stakeholder-frustration
- high-technical-debt
layout: solution
lang: de
en_slug: baseline-measurement
related_solutions:
- slug: delivery-performance-metrics
  similarity: 0.75
- slug: benefits-realization-tracking
  similarity: 0.7
- slug: outcome-based-goal-setting
  similarity: 0.65
- slug: quality-ratchet
  similarity: 0.65
- slug: fast-feedback-loops
  similarity: 0.65
- slug: improvement-budget
  similarity: 0.65
---

## Description

Baseline-Messung ist die Disziplin, den aktuellen Zustand — in Zahlen, bevor die Arbeit beginnt — von allem festzuhalten, was die Arbeit verbessern soll. Es ist die günstigste mögliche Intervention gegen den häufigsten Grund, warum technische Verbesserungen nicht gerechtfertigt werden können: nicht, dass ihre Vorteile unwirklich sind, sondern dass niemand sie demonstrieren kann, weil niemand aufgeschrieben hat, wie die Dinge vorher waren. Das Muster ist konsistent. Ein Team verbringt ein Quartal damit, Build-Zeiten, Vorfallraten oder manuellen Aufwand zu verringern, erreicht eine echte Verbesserung und kann dann nicht sagen, um wie viel, weil das „Vorher" nur als gemeinsamer Eindruck existiert. Der nächste Vorschlag trifft dann auf dieselbe Skepsis wie der letzte, und der Zyklus setzt sich fort. Zuerst zu messen kostet Tage; zuerst nicht zu messen kostet die Glaubwürdigkeit, die alles Nachfolgende finanziert.

## How to Apply ◆

> Die Messung muss üblicherweise geschehen, bevor irgendjemand zugestimmt hat, die Arbeit zu finanzieren, was bedeutet, dass das Team es spekulativ tun muss — und genau das ist der Grund, warum es nicht geschieht.

- **Entscheiden Sie, was die Arbeit ändern soll**, in einem Satz, bevor Sie wählen, was gemessen wird. Verbesserungen, die das Maß nicht benennen können, das sie bewegen wollen, sind Verbesserungen, deren Wert nachträglich zu Recht bestritten wird.
- **Rekonstruieren Sie Geschichte, statt eine Uhr zu starten.** Versionskontrolle, Ticket-Systeme, Deployment-Protokolle und Vorfallaufzeichnungen enthalten üblicherweise genug, um sechs bis zwölf Monate rückwirkend zu rekonstruieren. Dies ist weit besser als eine Baseline, die heute beginnt, weil es den Trend ebenso wie das Niveau zeigt.
- **Erfassen Sie drei bis fünf Maße, nicht mehr**, und wählen Sie welche, die günstig zu wiederholen sind. Eine Baseline, die zwei Wochen zur Produktion braucht, wird einmal gemessen, und eine einzelne Messung ist keine Baseline — der Vergleich ist der Punkt.
- **Messen Sie die Verteilung, nicht nur den Durchschnitt.** Die mediane Build-Zeit und das fünfundneunzigste Perzentil erzählen unterschiedliche Geschichten, und Verbesserungen bewegen häufig eines ohne das andere. Beide aufzuzeichnen verhindert einen späteren Streit darüber, welches gezählt hat.
- **Beziehen Sie ein Maß ein, um das sich das Geschäft bereits kümmert**, selbst wenn es nur lose an die Arbeit gekoppelt ist: Zeit zur Erfüllung einer Anfrage, für Kunden sichtbare Fehlerrate, Stunden manuellen Aufwands in einer Abteilung. Eine rein technische Baseline beweist eine Verbesserung einem Publikum, das nie im Zweifel war.
- **Schreiben Sie die Bedingungen auf**, nicht nur die Zahlen: Teamgröße, Systemlast, der Release-Rhythmus zu dieser Zeit, alles Ungewöhnliche im Zeitraum. Baselines werden nachträglich mit der Begründung angegriffen, dass sich etwas anderes geändert hat, und die Aufzeichnung ist die Verteidigung.
- **Veröffentlichen Sie die Baseline, bevor die Arbeit beginnt**, idealerweise an die Menschen, die das Ergebnis beurteilen werden. Eine nachträglich produzierte Baseline lädt, wie ehrlich auch immer, zum Verdacht ein, dass sie gewählt wurde, um das Ergebnis zu schmeicheln.
- **Messen Sie zu vereinbarten Zeitpunkten erneut**, nicht nur am Ende. Zwischenmessungen fangen eine Intervention ab, die nicht funktioniert, während noch Zeit ist, den Kurs zu ändern, was mehr wert ist als der eventuelle Beweis.
- **Berichten Sie ehrlich, wenn sich die Zahl nicht bewegt hat.** Ein Team, das seine Misserfolge berichtet, wird geglaubt, wenn es seine Erfolge berichtet, und dies ist der gesamte Mechanismus, durch den Messung die Glaubwürdigkeit aufbaut, von der spätere Vorschläge abhängen.

## Tradeoffs ⇄

> Baselines sind günstig und machen Nutzen beweisbar, aber sie kosten Aufwand, bevor irgendetwas finanziert wird, und sie schaffen die Möglichkeit, sichtbar falsch zu liegen.

**Vorteile:**

- Nutzen wird demonstrierbar statt behauptet, was der Unterschied zwischen einem Vorschlag ist, dem beim nächsten Mal geglaubt wird, und einem, dem nicht geglaubt wird.
- Zwischenmessung fängt unwirksame Interventionen früh ab, bevor die vollständige Investition in einen nicht funktionierenden Ansatz ausgegeben wird.
- Der Trend aus rekonstruierter Geschichte ist oft ein stärkeres Argument als das Niveau, da er zeigt, wohin sich die Situation entwickelt.
- Die Zuschreibung von Verbesserung zu einem spezifischen Arbeitsstück wird möglich, was einen einmaligen Genehmigungsvorgang in laufende Finanzierung verwandelt.
- Der Akt der Wahl der Maße erzwingt Klarheit darüber, wofür die Arbeit tatsächlich ist, was häufig ändert, was das Team zu tun beschließt.

**Kosten und Risiken:**

- Es erfordert Aufwand, bevor irgendetwas genehmigt wird, und dieser Aufwand wird nicht kompensiert, wenn der Vorschlag abgelehnt wird.
- Eine Baseline schafft die Möglichkeit zu demonstrieren, dass eine Verbesserung nicht funktioniert hat, was ein echtes Risiko für das Team ist, das sie produziert hat, und ein starker Anreiz, nicht zu messen.
- Maße werden zu Zielen. Alles, was zur Beurteilung von Erfolg genutzt wird, wird schließlich direkt optimiert, manchmal auf Kosten dessen, wofür es stand.
- Zuschreibung ist bestreitbar: andere Dinge ändern sich im selben Zeitraum, und ein entschlossener Skeptiker kann immer eine alternative Erklärung vorschlagen.
- Ein schlecht gewähltes Maß sperrt die Arbeit darauf fest, das Falsche zu optimieren, und das Maß mitten im Prozess zu ändern sieht wie das Verschieben der Torpfosten aus, selbst wenn es korrekt ist.

## How It Could Be

Ein Team verbrachte ein Quartal damit, seinen Build- und Testzyklus zu verringern, und berichtete danach, dass es „jetzt viel schneller" sei. Auf die Frage, um wie viel schneller, konnten sie es nicht sagen — die vorherige Dauer existierte nur als gemeinsame Erinnerung an „etwa eine halbe Stunde". Die Verbesserung, die substanziell war, produzierte keine Änderung darin, wie die Organisation solche Arbeit finanzierte. Vor der nächsten Anstrengung verbrachten sie zwei Tage damit, zwölf Monate an Pipeline-Dauern aus ihrem CI-System zu rekonstruieren: Median 31 Minuten, ansteigend auf 38 über das Jahr, fünfundneunzigstes Perzentil 74 Minuten. Sie zeichneten außerdem zwei geschäftsorientierte Maße auf: die mediane Zeit von der Meldung eines Fehlers bis zum Erreichen eines Fixes in Produktion, und die Anzahl der pro Monat deployten Änderungen. Nach der Arbeit wurden dieselben vier Zahlen erneut gemessen. Mediane Pipeline-Dauer 8 Minuten, fünfundneunzigstes Perzentil 14, Fehler-zu-Produktion-Zeit von 9 Tagen auf 3 gesunken, Deployments um 60 Prozent gestiegen. Die letzten beiden sicherten das Verbesserungsbudget des folgenden Jahres; die ersten beiden erklärten sie nur.

Die Ehrlichkeitsregel wurde ein Jahr später getestet. Das Team investierte sechs Wochen in eine Caching-Schicht, von der erwartet wurde, eine Berichtsgenerierungszeit zu verkürzen, über die sich eine Geschäftsabteilung ständig beschwerte. Die Baseline sagte Median 42 Sekunden, fünfundneunzigstes Perzentil 6 Minuten. Nach der Arbeit: Median 4 Sekunden, fünfundneunzigstes Perzentil 5 Minuten 40. Der Ausreißer — der das war, worüber sich die Abteilung tatsächlich beschwerte — hatte sich kaum bewegt, weil er von einer Abfrage dominiert wurde, die der Cache nicht berührte. Das Team berichtete dies offen, statt mit dem Median zu führen. Die unmittelbaren Kosten waren ein unangenehmes Meeting. Der bleibende Effekt war, dass acht Monate später, als dasselbe Team sagte, eine vorgeschlagene Änderung würde eine spezifische Verbesserung liefern, die Zahl ohne Widerspruch akzeptiert wurde.
