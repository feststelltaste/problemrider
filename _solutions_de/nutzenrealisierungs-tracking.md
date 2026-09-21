---
title: Nutzenrealisierungs-Tracking
description: Rückblick nach Abschluss der Arbeit, um zu berichten, ob der versprochene
  Nutzen tatsächlich eingetreten ist — auch wenn das nicht der Fall war.
category:
- Business
- Management
- Process
problems:
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- planning-credibility-issues
- stakeholder-confidence-loss
- short-term-focus
- feature-factory
- wasted-development-effort
- declining-business-metrics
- invisible-nature-of-technical-debt
- resource-waste
- delayed-value-delivery
- inability-to-innovate
- budget-overruns
- feature-bloat
- feedback-isolation
- increased-cost-of-development
- quality-degradation
- stakeholder-frustration
layout: solution
lang: de
en_slug: benefits-realization-tracking
related_solutions:
- slug: baseline-measurement
  similarity: 0.7
- slug: total-cost-of-ownership-transparency
  similarity: 0.65
- slug: subject-matter-reviews
  similarity: 0.65
- slug: staged-investment-with-decision-gates
  similarity: 0.65
- slug: cost-of-delay
  similarity: 0.65
- slug: blameless-postmortems
  similarity: 0.65
---

## Description

Nutzenrealisierungs-Tracking ist die Praxis, nach ausreichend verstrichener Zeit zu einer abgeschlossenen Investition zurückzukehren und, gegen die Zahlen, die sie rechtfertigten, zu berichten, was tatsächlich geschah. Fast keine Organisation tut dies. Business Cases werden vor der Genehmigung intensiv geprüft und nachher nie untersucht, was eine spezifische und schädliche Konsequenz hat: Niemand lernt, welche Arten von Behauptungen sich als wahr herausstellen. Die Schätzungen des technischen Teams werden unabhängig von ihrer Erfolgsbilanz mit derselben generischen Skepsis behandelt, weil es keine Erfolgsbilanz gibt. Nutzen-Tracking geht daher weniger um Verantwortlichkeit als um den Aufbau der Beweisgrundlage, die den nächsten Vorschlag glaubwürdig macht. Es ist außerdem der einzige Mechanismus, der zuverlässig die Investitionen erkennt, die still nicht funktioniert haben, welche sonst unbegrenzt weiterhin als Erfolge zitiert werden.

## How to Apply ◆

> Die Maße in einem Legacy-Business-Case brauchen üblicherweise ein Jahr oder mehr, um sich zu bewegen, was genau der Grund ist, warum niemand mehr hinschaut, wenn sie es tun.

- **Erfassen Sie die Behauptung zum Zeitpunkt der Genehmigung**, in der Form, in der sie später geprüft wird: welches Maß, um wie viel, bis wann. Ein Business Case, dessen Nutzen als „verbesserte Wartbarkeit" formuliert ist, kann nicht verifiziert werden, und diese Zweideutigkeit ist häufig auf beiden Seiten beabsichtigt.
- **Planen Sie die Überprüfung, wenn die Investition genehmigt wird**, nicht wenn sie abgeschlossen ist. Eine ungeplante Überprüfung geschieht nicht, und ein zum Genehmigungszeitpunkt gesetztes Datum ist weit schwieriger still fallen zu lassen als eines, das nachträglich vorgeschlagen wird.
- **Erlauben Sie ein realistisches Intervall.** Die Überprüfung drei Monate nach Abschluss einer Modernisierung misst Störung, nicht Nutzen. Zwölf Monate sind typisch für Legacy-Arbeit, mit einer Zwischenprüfung nach sechs.
- **Vergleichen Sie gegen die Baseline, die die Ausgabe rechtfertigte**, nicht gegen eine frische Messung des aktuellen Zustands. Wenn keine Baseline aufgezeichnet wurde, sagen Sie das offen — dieser Befund ist selbst das Argument dafür, beim nächsten Mal eine aufzuzeichnen.
- **Berichten Sie die Fehlschläge so prominent wie die Treffer.** Eine Tracking-Praxis, die nur Erfolge zutage bringt, ist Marketing, und jeder erkennt das innerhalb von zwei Zyklen als solches. Die Glaubwürdigkeit, die die Praxis lohnend macht, kommt vollständig aus ihrer Bereitschaft, Fehlschläge zu berichten.
- **Trennen Sie „der Nutzen ist nicht eingetreten" von „die Arbeit wurde nicht getan".** Diese haben unterschiedliche Lehren: Das erste sagt, die Theorie war falsch, das zweite sagt, die Ausführung war es. Diese zu vermengen verhindert, dass eines von beiden gelernt wird.
- **Suchen Sie nach Nutzen, der nicht vorhergesagt wurde.** Legacy-Verbesserungen produzieren routinemäßig Effekte, die niemand behauptet hat — ein stillgelegtes System, das eine unabhängige Lizenz freigab, eine Refaktorierung, die ein ungeplantes Feature günstig machte. Diese zu erfassen verbessert die Genauigkeit zukünftiger Schätzungen, die üblicherweise unter- statt überschätzen.
- **Führen Sie ein laufendes Protokoll über Investitionen hinweg.** Das Muster über zehn Überprüfungen — welche Kategorien von Behauptungen sich bewahrheiten, welche konsistent optimistisch sind — ist weit wertvoller als jede einzelne Überprüfung und ist es, was schließlich ändert, wie Vorschläge aufgenommen werden.
- **Halten Sie die Überprüfung günstig.** Ein halber Tag gegen drei oder vier Maße. Ein schwergewichtiger Prozess wird übersprungen, und eine übersprungene Überprüfung liefert überhaupt keine Beweise.

## Tradeoffs ⇄

> Nutzen-Tracking ist, was zukünftige Vorschläge glaubwürdig macht, aber es schafft ein Protokoll von Fehlschlägen und erfordert, dass sich jemand ein Jahr später noch kümmert.

**Vorteile:**

- Vorschläge von einem Team mit nachgewiesener Erfolgsbilanz werden anders aufgenommen, was sich über die Zeit zu erheblich einfacherer Finanzierung für technische Arbeit summiert.
- Schätzung verbessert sich, weil die Organisation lernt, welche Kategorien von Behauptungen systematisch optimistisch sind und um ungefähr wie viel.
- Investitionen, die still nicht funktioniert haben, werden erkannt, statt jahrelang als Erfolge zitiert und genutzt zu werden, um die Wiederholung des Ansatzes zu rechtfertigen.
- Unvorhergesagter Nutzen wird erfasst, und dieser ist in Legacy-Arbeit häufig substanziell, wo Effekte zweiter Ordnung schwer vorherzusehen sind.
- Das Wissen, dass eine Überprüfung stattfinden wird, verbessert die Ehrlichkeit von Business Cases zum Zeitpunkt ihrer Erstellung, was der größte Effekt sein könnte.

**Kosten und Risiken:**

- Es produziert dokumentierte Fehlschläge, was politisch unangenehm ist und einen starken Anreiz schafft, die Praxis auslaufen zu lassen.
- Zuschreibung ein Jahr später ist genuin schwierig: mehrere Dinge haben sich im Intervall geändert, und sowohl Anerkennung als auch Schuld sind bestreitbar.
- Die Menschen, die die Arbeit genehmigt und geliefert haben, sind oft weitergezogen, sodass niemand mit dem Kontext oder der Motivation zurückbleibt, die Überprüfung durchzuführen.
- Punitiv genutzt, macht es zukünftige Business Cases konservativ und vage, was das Gegenteil des beabsichtigten Effekts ist.
- Zwölfmonatige Intervalle passen unbequem zu jährlichen Planungszyklen, sodass der Befund oft nach der Entscheidung ankommt, die er hätte informieren sollen.

## How It Could Be

Eine Organisation hatte über vier Jahre elf technische Investitionen genehmigt und keine davon überprüft. Ihr Engineering-Direktor führte eine Regel ein: Jede genehmigte Investition über einer Schwelle trug eine geplante zwölfmonatige Überprüfung, ein halber Tag, gegen die Maße im ursprünglichen Fall. Die ersten vier Überprüfungen waren unangenehm. Zwei Investitionen hatten ungefähr das geliefert, was behauptet wurde. Eine hatte etwa ein Drittel davon geliefert — eine Testautomatisierungsanstrengung, deren versprochene Verringerung des manuellen Testaufwands durch eine gleichzeitige Umfangserweiterung aufgezehrt worden war, die niemand berücksichtigt hatte. Eine hatte überhaupt nichts Messbares geliefert, weil das im Business Case genannte Maß nie instrumentiert worden war und nicht rekonstruiert werden konnte. Der letzte Befund änderte die Praxis mehr als die anderen: Baseline-Instrumentierung wurde zur Voraussetzung für Genehmigung.

Bis zur neunten Überprüfung war das Muster zum nützlichen Ergebnis geworden. Behauptungen über Vorfallreduktion hatten sich gut bewahrheitet, im Durchschnitt etwa 80 Prozent dessen, was projiziert wurde. Behauptungen über Entwicklerproduktivität hatten sich schlecht bewahrheitet, im Durchschnitt unter 30 Prozent, konsistent weil die projizierten Zeitersparnisse von anderer Arbeit absorbiert statt in Durchsatz umgewandelt wurden. Behauptungen über Lizenz- und Infrastruktureinsparungen waren fast genau richtig gewesen, da sie am einfachsten zu schätzen waren. Die Organisation hörte nicht auf, Produktivitätsverbesserungen zu finanzieren — sie begann, diese Behauptungen um einen genannten Faktor zu diskontieren und zu verlangen, dass der Fall zur diskontierten Zahl funktioniert. Zwei Vorschläge, die zuvor genehmigt worden wären, wurden auf dieser Basis abgelehnt, und einer, der zweimal abgelehnt worden war, wurde genehmigt, sobald sein Autor den Nutzen als Vorfallreduktion umformulierte, was das Protokoll als die Behauptungsart zeigte, die sich tatsächlich bewahrheitete.
