---
title: Schuldenklassifizierung
description: Sortierung technischer Schulden danach, ob sie tatsächlich etwas kosten,
  sodass Aufwand in die Schulden fließt, die kosten, und der Rest ohne schlechtes
  Gewissen liegen bleiben kann.
category:
- Code
- Process
- Management
problems:
- high-technical-debt
- invisible-nature-of-technical-debt
- difficulty-quantifying-benefits
- maintenance-paralysis
- modernization-strategy-paralysis
- perfectionist-culture
- accumulation-of-workarounds
- increasing-brittleness
- brittle-codebase
- competing-priorities
- short-term-focus
- refactoring-avoidance
- analysis-paralysis
- increased-technical-shortcuts
- quality-compromises
- workaround-culture
- low-code-customization-sprawl
layout: solution
lang: de
en_slug: debt-classification
related_solutions:
- slug: technical-debt-backlog
  similarity: 0.85
- slug: debt-accrual-analysis
  similarity: 0.8
- slug: debt-remediation-estimation
  similarity: 0.8
- slug: technical-debt-assessment
  similarity: 0.75
- slug: functional-debt-management
  similarity: 0.75
- slug: code-hotspot-analysis
  similarity: 0.7
---

## Description

Schuldenklassifizierung sortiert die bekannten technischen Schulden danach, ob sie tatsächlich etwas kosten, statt danach, wie unangenehm sie sind. Die zentrale Unterscheidung ist zwischen Schulden, die Zinsen tragen, und Schulden, die ruhen: Ein schlecht strukturiertes Modul, das drei Personen jede Woche modifizieren, kostet kontinuierlich echtes Geld, während ein ebenso schlechtes Modul, das seit vier Jahren niemand angefasst hat, nichts kostet und nichts kosten wird, bis jemand es anfasst. Teams treffen diese Unterscheidung nicht natürlicherweise, weil die emotionale Reaktion auf schlechten Code davon getrieben wird, wie er sich liest, statt davon, was er kostet. Das Ergebnis ist Aufwand, der auf den anstößigsten Code statt auf den teuersten verwendet wird, und ein durchdringendes Gefühl, dass das gesamte System eine Belastung ist. Klassifizierung ist das, was die Schulden proportional macht — und der Großteil davon stellt sich bei näherer Betrachtung als unwichtig heraus.

## How to Apply ◆

> Ein Legacy-System enthält eine enorme Menge an Schulden, die nie bezahlt werden, weil sich nie jemand in ihre Nähe begeben wird, und diesen Anteil zu identifizieren ist genauso wertvoll wie den Rest zu identifizieren.

- **Stellen Sie fest, ob jeder Posten zinstragend ist.** Der Test ist empirisch, nicht ästhetisch: Wurde dieser Code im letzten Jahr geändert, ist er an Vorfällen beteiligt, verlangsamt er Arbeit, die Menschen tatsächlich tun? Änderungshäufigkeit aus der Versionskontrolle beantwortet das meiste davon an einem Nachmittag.
- **Trennen Sie bewusste von unbeabsichtigten Schulden.** Schulden, die wissentlich unter Zeitdruck aufgenommen wurden, mit einem Grund, sind ein anderes Managementproblem als Schulden, die sich angehäuft haben, weil niemand es besser wusste. Erstere brauchen eine Rückzahlungsentscheidung; zweitere brauchen einen Kompetenzeingriff, und beide gleich zu behandeln adressiert keines von beidem.
- **Unterscheiden Sie Schulden, die blockieren, von Schulden, die verlangsamen.** Etwas, das eine Änderung unmöglich macht oder eine ganze Klasse von Arbeit unerreichbar macht, rangiert über etwas, das jede Änderung fünfzehn Prozent mühsamer macht — selbst wenn Letzteres verbreiteter und ärgerlicher ist.
- **Markieren Sie ruhende Schulden explizit als akzeptiert**, schriftlich, statt sie in einem Backlog liegen zu lassen. Ein Posten, der wissentlich nicht angegangen wird, sollte das sagen und sagen warum. Dies ist der Schritt, der die Liste auf etwas schrumpft, das ein Team ohne Verzweiflung betrachten kann.
- **Klassifizieren Sie neu, wenn sich Umstände ändern.** Ruhende Schulden werden zinstragend in dem Moment, in dem ein Roadmap-Punkt diesen Bereich berührt, sodass die Klassifizierung überarbeitet werden sollte, wenn sich Pläne ändern, nicht jährlich.
- **Nutzen Sie die Klassifizierung, um die Reaktion festzulegen**, nicht nur die Reihenfolge. Zinstragende und blockierende Schulden werden behoben; zinstragende und verlangsamende Schulden werden opportunistisch durch vorbereitendes Refactoring adressiert; ruhende Schulden werden hinter einer Schnittstelle eingegrenzt oder liegen gelassen; und Schulden in zur Löschung geplantem Code bekommen nichts.
- **Erfassen Sie die Begründung pro Posten**, kurz. Klassifizierung ohne Gründe wird jedes Mal angefochten, wenn jemand Neues auf die Liste schaut, und die Gründe sind das, was einen Nachfolger sinnvoll neu klassifizieren lässt.
- **Berichten Sie das Profil, nicht nur die Summe.** „Wir haben 140 Schuldenposten" ist beängstigend und nutzlos. „Von 140 Posten sind 22 zinstragend, 6 davon blockierend, und 118 sind ruhend und akzeptiert" ist eine Managementaussage.
- **Achten Sie auf den ästhetischen Reflex.** Die stärkste Fürsprache haftet meist an den Schulden, die am unangenehmsten zu lesen sind, und diese Korrelation mit tatsächlichen Kosten ist schwach. Belege für die zinstragende Klassifizierung zu verlangen ist das, was dies ehrlich hält.

## Tradeoffs ⇄

> Klassifizierung lenkt Aufwand auf die Schulden, die kosten, und macht den Rest explizit akzeptabel, erfordert aber Ermessensentscheidungen, die manchmal falsch sein werden und die genutzt werden können, um echte Probleme abzutun.

**Vorteile:**

- Der Aufwand konzentriert sich auf Schulden, die tatsächlich kosten, was typischerweise ein kleiner Bruchteil dessen ist, was ein Team als Schulden wahrnimmt.
- Die Liste wird begrenzt und überprüfbar, weil die ruhende Mehrheit explizit akzeptiert ist, statt als dauerhaftes unerledigtes Geschäft dazuliegen.
- Die Furcht wird proportional. Ein Großteil der Angst vor einem Legacy-System kommt daher, alle seine Mängel als gleich lebendig zu behandeln, und das sind sie nicht.
- Unterschiedliche Schuldentypen erhalten unterschiedliche Reaktionen, was effizienter ist, als alles als Refactoring-Kandidat zu behandeln.
- Das Profil ist gegenüber dem Management auf eine Weise kommunizierbar, wie es eine rohe Zahl nicht ist, was Abhilfeanfragen glaubwürdig macht.

**Kosten und Risiken:**

- Ruhende Schulden werden gelegentlich ohne Vorwarnung dringend — eine Sicherheitswarnung, eine unerwartete Feature-Anfrage —, und als unantastbar akzeptierter Code kann sich als anfassungsbedürftig herausstellen.
- Die Klassifizierung erfordert Ermessen, und ein Team unter Druck wird unbequem teure Posten als ruhend einstufen.
- Schulden explizit zu akzeptieren kann als Tolerierung schlechter Qualität gelesen werden, und es muss als Priorisierungsentscheidung gerahmt werden, nicht als Standard.
- Änderungshäufigkeit ist ein Proxy. Code, der gerade deshalb gemieden wird, weil er beängstigend ist, sieht in den Daten ruhend aus, während er eine ernste Belastung ist.
- Neuklassifizierung wird leicht übersprungen, was eine veraltete Klassifizierung hinterlässt, die etwas als ruhend bezeichnet, obwohl die Roadmap gerade darauf zielt.

## How It Could Be

Ein Team, das ein Fertigungssystem pflegte, hatte einen Backlog technischer Schulden mit 187 Posten, angehäuft über sechs Jahre, den niemand anschaute, weil das Hinschauen demoralisierend war. Sie klassifizierten ihn über drei Tage mittels Änderungshäufigkeit aus der Versionskontrolle und ihrer Vorfallaufzeichnung. Neunundzwanzig Posten waren zinstragend. Fünf davon waren blockierend — sie machten spezifische geplante Arbeit unmöglich statt bloß schwerer. Die verbleibenden 158 waren ruhend, und 41 davon lagen in Code, der seit 2018 nicht modifiziert worden war. Die 158 wurden mit jeweils einer einzeiligen Begründung als akzeptiert markiert und aus der aktiven Liste entfernt. Die verbleibende 29-Posten-Liste war zum ersten Mal etwas, das das Team bei der Planung anschaute. Drei der fünf blockierenden Posten wurden im folgenden Quartal behoben.

Die Trennung bewusst-versus-unbeabsichtigt veränderte ein Managementgespräch. Die 29 zinstragenden Posten nach Herkunft zu klassifizieren zeigte, dass 19 bewusste Abkürzungen waren, genommen unter bestimmten Fristen, alle zum damaligen Zeitpunkt markiert und keine je überarbeitet. Das war kein Codequalitätsproblem — es war ein Prozessproblem, und die Korrektur war eine Regel, dass jede unter Termindruck genommene Abkürzung eine verpflichtende Überprüfung im nächsten Quartal trug. Die anderen 10 waren unbeabsichtigt, konzentriert auf Arbeit zweier Entwickler während einer Periode, in der das Team keine nennenswerte Review-Praxis hatte. Das war ein Kompetenzproblem, und es wurde durch Coaching statt durch Refactoring adressiert. Keine der beiden Reaktionen wäre aus einer Liste gewählt worden, die alle 29 Posten als dieselbe Art von Ding behandelte.
