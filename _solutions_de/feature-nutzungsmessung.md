---
title: Feature-Nutzungsmessung
description: Instrumentierung des Systems, um zu erfassen, welche Features tatsächlich
  von wem genutzt werden, sodass Wartungsaufwand und Löschentscheidungen auf Evidenz
  beruhen.
category:
- Business
- Process
- Requirements
problems:
- gold-plating
- feature-creep
- feature-factory
- high-maintenance-costs
- code-duplication
- delayed-value-delivery
- system-stagnation
- resource-waste
- wasted-development-effort
- increased-cost-of-development
- reduced-innovation
- product-direction-chaos
- budget-overruns
- duplicated-work
- maintenance-cost-increase
- project-resource-constraints
- competing-priorities
- declining-business-metrics
- market-pressure
- modernization-roi-justification-failure
- short-term-focus
- difficulty-quantifying-benefits
- feature-bloat
- excessive-customization
- custom-report-sprawl
- reimplemented-standard-functionality
layout: solution
lang: de
en_slug: feature-usage-measurement
related_solutions:
- slug: deprecation-strategy
  similarity: 0.7
- slug: total-cost-of-ownership-transparency
  similarity: 0.7
- slug: system-decommissioning
  similarity: 0.7
- slug: code-metrics
  similarity: 0.7
- slug: feature-flags
  similarity: 0.7
- slug: technical-debt-backlog
  similarity: 0.65
---

## Description

Feature-Nutzungsmessung ist die Instrumentierung eines Systems, um zu erfassen, welche seiner Fähigkeiten tatsächlich genutzt werden, wie oft und von welchen Nutzerarten. Sie beantwortet eine Frage, die Legacy-Organisationen üblicherweise überhaupt nicht beantworten können: Von allem, was dieses System tut, was zählt. Die Abwesenheit dieser Antwort hat zwei teure Konsequenzen. Wartungsaufwand wird gleichmäßig über Features verteilt, unabhängig von ihrem Wert, sodass der von vier Personen im Jahr genutzte Codepfad denselben Schutz erhält wie der ständig genutzte. Und nichts wird jemals entfernt, weil Entfernung erfordert, dass jemand behauptet, ein Feature sei ungenutzt, und ohne Daten wird niemand dieses Risiko eingehen. Jedes nicht entfernte Feature ist dauerhaftes Gewicht: Code, der gepflegt werden muss, Tests, die bestehen bleiben müssen, und eine Einschränkung für jede zukünftige Änderung. Messung verwandelt Löschung von einem Glücksspiel in eine Entscheidung.

## How to Apply ◆

> Legacy-Systeme sammeln über Jahrzehnte von Anfragen Features an, und die übliche Schätzung, dass ein beträchtlicher Anteil selten oder nie genutzt wird, bestätigt sich fast immer, sobald jemand endlich misst.

- **Instrumentieren Sie auf der Ebene nutzerbedeutsamer Fähigkeit**, nicht auf der Ebene von Funktionsaufrufen. „Wie viele Nutzer haben diesen Quartal einen benutzerdefinierten Bericht erzeugt" ist handlungsleitend; eine Aufrufzählung für eine interne Methode ist es nicht.
- **Erfassen Sie, wer es nutzt, nicht nur wie oft.** Ein zweimal jährlich vom Regulator genutztes Feature ist kein Kandidat für Löschung, und ein ständig von einem abwandernden Kunden genutztes Feature ist ein anderes Risiko. Aggregierte Zählungen verbergen beides.
- **Messen Sie über einen vollständigen Geschäftszyklus**, bevor Sie Schlüsse ziehen. Jahresberichts-Features, Jahresendprozesse und saisonale Fähigkeiten sind in einem Quartal Daten unsichtbar, und eines davon zu löschen ist der Fehler, der die gesamte Praxis diskreditiert.
- **Beginnen Sie mit den Bereichen, in denen Sie die Antwort vermuten**, statt alles zu instrumentieren. Die Features, die teuer zu warten und vermutlich marginal sind, sind dort, wo sich die Messung zuerst rechnet.
- Kombinieren Sie Nutzung mit **Wartungskosten** — Änderungshäufigkeit, Defektzahl, Vorfallbeteiligung. Die Löschkandidaten sind die Schnittmenge von selten genutzt und teuer zu erhalten, und diese Schnittmenge ist meist klein, offensichtlich und handlungswürdig.
- **Ankündigen und beobachten vor dem Entfernen.** Markieren Sie das Feature als veraltet, benachrichtigen Sie identifizierbare Nutzer, und instrumentieren Sie dann den Zeitraum nach der Ankündigung. Sich allein auf die Daten zu verlassen findet gelegentlich den einen Nutzer, auf den es ankommt, auf unangenehme Weise.
- **Entfernen Sie auf reversible Weise**: Zunächst hinter einem Flag deaktivieren, einen vollen Zyklus warten, dann den Code löschen. Die Lücke zwischen Deaktivieren und Löschen ist, wo sich die ungemessenen Nutzer melden.
- Nutzen Sie die Daten, um **sowohl den Aufbau als auch die Entfernung zu informieren**. Ein Team, das Features ausliefert und nie erfährt, ob sie genutzt werden, hat überhaupt keine Feedback-Schleife, und dies ist der Mechanismus, durch den sich eine Feature Factory selbst erhält.
- **Respektieren Sie Datenschutzbeschränkungen** bei dem, was erfasst wird. Nutzungsmessung muss selten Einzelpersonen identifizieren; aggregieren Sie nach Rolle, Mandant oder Segment, und beziehen Sie die Datenschutzfunktion ein, bevor Sie irgendetwas Nutzersichtbares instrumentieren.

## Tradeoffs ⇄

> Nutzungsdaten ermöglichen Löschung und lenken Wartungsaufwand, aber Instrumentierung ist Arbeit, die Daten sind leicht misszuverstehen, und ein Feature zu entfernen ist unumkehrbar auf eine Weise, wie es das Behalten nicht ist.

**Vorteile:**

- Löschung wird möglich, und Löschung ist die einzige Intervention, die die Größe und Komplexität eines Systems absolut reduziert statt sie umzuorganisieren.
- Wartungs- und Testaufwand kann auf das konzentriert werden, was tatsächlich genutzt wird, was in einem System mit einem langen Schwanz marginaler Features eine beträchtliche Umverteilung darstellt.
- Anfragen nach neuen Features können anhand von Evidenz darüber diskutiert werden, wie vergleichbare frühere Features abgeschnitten haben, was das stärkste verfügbare Argument gegen den Bau auf Spekulation ist.
- Der vom Team gelieferte Wert wird in anderen Begriffen als Ausstoßvolumen sichtbar, was verändert, wie über eine Feature Factory diskutiert wird.
- Der Modernisierungsumfang schrumpft. Als ungenutzt bestätigte Features müssen nicht migriert werden, und dies entfernt häufig einen bedeutenden Anteil der Arbeit.

**Kosten und Risiken:**

- Instrumentierung muss gebaut und gepflegt werden, und ein Legacy-System ohne Telemetrie-Framework zu instrumentieren ist keine kleine Aufgabe.
- Geringe Nutzung ist nicht dasselbe wie geringer Wert. Regulatorische, vertragliche und Disaster-Recovery-Fähigkeiten mögen fast nie ausgeübt werden und trotzdem unverzichtbar sein.
- Das Entfernen eines Features ist in der Praxis unumkehrbar, und eine falsche Entfernung schadet dem Vertrauen in die Daten und in das Team weit mehr, als ein nicht entferntes Feature kostet.
- Messfenster, die jährliche oder saisonale Zyklen verfehlen, produzieren selbstsicher falsche Schlussfolgerungen.
- Nutzungsverfolgung wirft legitime Datenschutzbedenken auf und kann rechtliche Prüfung, Einwilligung oder Beschränkung auf aggregierte Daten erfordern.

## How It Could Be

Ein Team, das ein betriebliches Spesenmanagementsystem pflegte, trug 340 verschiedene, über vierzehn Jahre angesammelte Fähigkeiten, und jede Modernisierungsschätzung brach unter dem Gewicht zusammen, alle davon zu migrieren. Sie instrumentierten die übergeordneten Nutzeraktionen und maßen dreizehn Monate lang, um einen vollständigen Geschäftsjahreszyklus abzudecken. Das Ergebnis: 61 Fähigkeiten hatten null erfasste Nutzung, und weitere 88 wurden von weniger als fünf Personen im gesamten Jahr genutzt. Die Kreuzung mit Wartungsdaten zeigte, dass 30 der ungenutzten Fähigkeiten in Modulen mit hoher Änderungshäufigkeit saßen, was bedeutete, dass Entwickler wiederholt Code pflegten, den nichts ausübte. Nach einer Veraltungsankündigung und einem weiteren Quartal Beobachtung wurden 54 Fähigkeiten entfernt. Zwei erzeugten Beschwerden, beide von einem einzigen Finanznutzer, und beide wurden innerhalb eines Tages wiederhergestellt. Der Migrationsumfang schrumpfte um etwa ein Fünftel.

Die Messung veränderte auch eine Entscheidung darüber, was gebaut werden sollte. Das Team hatte eine langjährige Anfrage nach einer ausgefeilteren Genehmigungs-Routing-Engine, gerechtfertigt durch die Behauptung, dass die bestehenden einfachen Regeln unzureichend seien. Die Nutzungsdaten zeigten, dass 94 Prozent der Spesenabrechnungen einem von drei Routing-Pfaden folgten und dass die flexiblen Optionen in der aktuellen Engine — fünf Jahre zuvor aus demselben Grund hinzugefügt — von zwei von vierzig Abteilungen konfiguriert worden waren. Statt die ausgefeiltere Engine zu bauen, vereinfachte das Team die bestehende um die drei dominanten Pfade herum und behandelte den Rest manuell. Der Wartungsaufwand des Routing-Subsystems fiel erheblich, und die Feature-Anfrage wurde zurückgezogen, sobald ihr Sponsor die Verteilung sah.
