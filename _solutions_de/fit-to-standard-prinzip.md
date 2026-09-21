---
title: Fit-to-Standard-Prinzip
description: Die Übernahme des eigenen Prozesses des Produkts zum Standard machen
  und verlangen, dass jede Abweichung von jemandem begründet, bemessen und genehmigt
  wird, der ihre Kosten trägt.
category:
- Business
- Process
- Architecture
problems:
- process-software-misfit
- reimplemented-standard-functionality
- excessive-customization
- core-modification-of-standard-software
- upgrade-blocked-by-customization
- inefficient-processes
- inadequate-requirements-gathering
- eager-to-please-stakeholders
- increased-cost-of-development
- high-maintenance-costs
- gold-plating
- feature-creep
- voided-vendor-support
layout: solution
lang: de
en_slug: fit-to-standard-principle
related_solutions:
- slug: explicit-extension-points
  similarity: 0.7
- slug: variant-consolidation
  similarity: 0.7
- slug: customization-cost-attribution
  similarity: 0.65
- slug: standard-software
  similarity: 0.65
- slug: change-management-process
  similarity: 0.65
- slug: evolutionary-requirements-development
  similarity: 0.6
---

## Description

Das Fit-to-Standard-Prinzip kehrt den Standardfall bei der Einführung von Standardsoftware um: Der Prozess des Produkts wird übernommen, sofern es keinen erklärten Grund dagegen gibt, statt dass das Produkt angepasst wird, sofern es keinen Grund dagegen gibt. Die Umkehrung ist wichtig, weil Standardfälle die Ergebnisse bestimmen, wenn niemand hinschaut. Unter der üblichen Regelung wird eine als aktuelle Praxis ausgedrückte Anforderung durch Trägheit zu einer Anpassung, und das kumulative Ergebnis ist ein stark angepasstes System, das die Vergangenheit reproduziert. Unter Fit-to-Standard muss dieselbe Anforderung eine Frage überstehen — was tut das Produkt hier, und warum reicht das nicht aus —, und ein beträchtlicher Anteil der Anforderungen übersteht sie nicht, weil sie Beschreibungen von Gewohnheit statt Aussagen über Bedarf waren.

## How to Apply ◆

> Die meisten Anforderungen in einer Paketsoftware-Einführung beschreiben, wie die Organisation heute arbeitet, und der Wert des Prinzips liegt darin, jemanden zu zwingen zu fragen, ob das erhaltenswert ist.

- **Etablieren Sie, was der Standard tut, bevor Sie Anforderungen sammeln**, nicht danach. Ein Workshop, der mit einer Demonstration des Produktprozesses beginnt, erzeugt ein anderes Set an Anforderungen als einer, der mit der Dokumentation des aktuellen Prozesses beginnt.
- **Verlangen Sie einen erklärten Grund für jede Abweichung**, in fester Form: was der Standard tut, warum das nicht ausreicht, was es das Unternehmen kosten würde, stattdessen den Prozess anzupassen, und wer die laufenden Kosten des Unterschieds akzeptiert.
- **Unterscheiden Sie echt differenzierende Prozesse von bloß gewohnheitsmäßigen.** Der Wettbewerbsvorteil einer Organisation ist eine Anpassung wert; ihre Kreditorenbuchhaltungs-Genehmigungsreihenfolge fast nie. Die meisten Abweichungsanfragen betreffen die zweite Kategorie.
- **Legen Sie die Genehmigung bei jemandem an, der die Konsequenz trägt.** Eine von der anfragenden Abteilung genehmigte Abweichung kostet sie nichts; genehmigt von demjenigen, der das Upgrade-Budget besitzt, ist es eine echte Entscheidung.
- **Hängen Sie die Lebenszeitkosten an die Anfrage**, nicht nur die Bau-Schätzung. Die Implementierung ist ein Bruchteil der Gesamtkosten, und nur diesen Bruchteil zu präsentieren garantiert systematische Unterschätzung jeder jemals vorgeschlagenen Anpassung.
- **Setzen Sie der Herausforderung eine Zeitbox.** Fit-to-Standard wird zur Behinderung, wenn jede Anfrage eine ausgedehnte Untersuchung auslöst. Eine definierte kurze Bewertung, mit einer Standardantwort, falls sie nicht abgeschlossen wird, hält es praktikabel.
- **Erfassen Sie die genehmigten Abweichungen** als gepflegtes Register mit Gründen, damit die angesammelte Menge später überprüft werden kann und künftige Upgrades wissen, was sie mit sich tragen.
- **Überprüfen Sie Abweichungen bei jedem größeren Release erneut.** Der Anbieter hat die Lücke vielleicht geschlossen, in welchem Fall die Abweichung stillgelegt werden kann — und niemand wird es bemerken, sofern nicht jemand nachprüft.
- **Geben Sie dem Prinzip einen benannten Eigentümer mit Autorität.** Ohne jemanden, der befugt ist, Nein zu sagen, kehrt der Standardfall bei den ersten paar umstrittenen Anfragen zur Entgegenkommen zurück.

## Tradeoffs ⇄

> Standardmäßig den Standard zu übernehmen bewahrt Upgradefähigkeit und entfernt eine große Menge unnötiger Anpassung, aber es erfordert, dass die Organisation ihre Arbeitsweise ändert und jemand mit Autorität darauf besteht.

**Vorteile:**

- Das Anpassungsvolumen sinkt erheblich, und mit ihm die Kosten jedes künftigen Upgrades und jeder künftigen Änderung.
- Anforderungen werden geprüft statt transkribiert, was häufig offenbart, dass ein beschriebener Bedarf eine Beschreibung von Gewohnheit war.
- Die Organisation erhält die Prozessverbesserungen des Anbieters automatisch, statt einen Fork einer älteren Arbeitsweise zu pflegen.
- Genehmigte Abweichungen werden mit Gründen dokumentiert, was die angesammelte Menge überprüfbar statt archäologisch macht.
- Die Implementierung ist schneller, weil die Konfiguration des Standards schneller geht als der Bau einer Alternative dazu.

**Kosten und Risiken:**

- Geschäftsprozesse müssen sich ändern, was disruptiv ist, auf Widerstand stößt und Autorität erfordert, die Softwareprojekte häufig nicht haben.
- Dogmatisch angewendet zwingt es echt differenzierende Prozesse in eine generische Form und kann einen echten Wettbewerbsvorteil beschädigen.
- Die Bewertung fügt jeder Anforderung Latenz hinzu, und wenn der Prozess schwerfällig ist, wird er zu einem Hindernis, um das Menschen herum navigieren.
- Der Standardprozess des Produkts ist nicht immer gut; Anbieter kodieren Annahmen, die möglicherweise nicht zu Ihrer Branche oder Ihrem Maßstab passen.
- Change Management für das betroffene Personal ist eine beträchtliche Kostenposition, die routinemäßig aus dem Vergleich weggelassen wird, was Fit-to-Standard günstiger aussehen lässt, als es ist.

## How It Could Be

Eine Organisation, die eine Dokumenten- und Aktenplattform einführte, führte Anforderungsworkshops durch, die mit einer Demonstration des Standardprozesses begannen, gefolgt von der Frage, wo er nicht funktionieren würde. Von 140 erhobenen Anforderungen wurden 96 durch Konfiguration erfüllt, sobald Teilnehmer gesehen hatten, was das Produkt tat. Von den verbleibenden 44 beseitigte das Abweichungsformular — was der Standard tut, warum das nicht ausreicht, was die Änderung des Prozesses kosten würde — weitere 19, in den meisten Fällen weil die anfragende Abteilung die zweite Frage mit nichts über die aktuelle Praxis hinaus beantworten konnte. Fünfundzwanzig Abweichungen wurden genehmigt, jede mit einem Grund und einem akzeptierenden Eigentümer erfasst. Eine vergleichbare Einführung bei einer Schwesterorganisation zwei Jahre zuvor, auf konventionelle Weise durchgeführt, hatte 130 Anpassungen produziert.

Die Wiedervorlage-Disziplin zahlte sich später und unerwartet aus. Beim zweiten größeren Release nach dem Go-Live wurde das Abweichungsregister gegen die Release Notes des Anbieters geprüft. Vier der 25 Abweichungen waren durch Standardfunktionalität geschlossen worden, die der Anbieter inzwischen ausgeliefert hatte, und eine war unnötig geworden, weil sich die Abteilung, die sie bediente, umstrukturiert hatte. Die Stilllegung dieser fünf dauerte drei Wochen. Ohne das Register wäre keine davon bemerkt worden — die Abweichungen wären einfach als Teil des Systems fortgeführt worden, wie die vierhundert bei ihrer Schwesterorganisation.
