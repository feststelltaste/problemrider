---
title: Analyse der Schuldenentstehung
description: Herausfinden, warum Schulden immer wieder an denselben Stellen entstehen,
  und den Mechanismus beheben — denn Schulden abzubauen, während die Entstehung
  weitergeht, ist ein Laufband.
category:
- Process
- Code
- Management
problems:
- high-technical-debt
- accumulation-of-workarounds
- increased-technical-shortcuts
- increasing-brittleness
- quality-degradation
- workaround-culture
- invisible-nature-of-technical-debt
- quality-compromises
- refactoring-avoidance
- copy-paste-programming
- inconsistent-execution
- maintenance-cost-increase
- convenience-driven-development
- short-term-focus
- code-duplication
layout: solution
lang: de
en_slug: debt-accrual-analysis
related_solutions:
- slug: technical-debt-backlog
  similarity: 0.8
- slug: debt-classification
  similarity: 0.8
- slug: technical-debt-assessment
  similarity: 0.8
- slug: debt-remediation-estimation
  similarity: 0.75
- slug: code-hotspot-analysis
  similarity: 0.7
- slug: blameless-postmortems
  similarity: 0.7
---

## Description

Analyse der Schuldenentstehung stellt eine andere Frage als die übliche. Statt „welche Schulden haben wir" fragt sie „was erzeugt sie immer wieder" und behandelt die Antwort als das, was zu beheben ist. Die Unterscheidung ist wichtig, weil Abbau ohne sie ein Laufband ist: Ein Team baut Schulden mit einer bestimmten Rate ab, während die Organisation sie mit einer anderen erzeugt, und wenn die zweite Rate höher ist, ist der Aufwand unsichtbar und die Moral des Teams gibt irgendwann nach. Die Ursachen sind selten mysteriös und fast nie ein Mangel an Fähigkeit oder Sorgfalt. Sie sind strukturell — Termindruck, angewandt an einem bestimmten Punkt im Zyklus, eine fehlende Testfähigkeit, die den sicheren Pfad teuer macht, eine Eigentümerschaftslücke, bei der niemand zuständig ist, eine Review-Praxis, die eine bestimmte Klasse von Problemen nicht erkennt. Jede ist adressierbar, und die Adressierung einer davon verhindert typischerweise mehr Schulden, als Monate von Abbau entfernen.

## How to Apply ◆

> Legacy-Schulden sind meist kein einzelnes historisches Ereignis, sondern ein laufender Prozess, und der Prozess ist normalerweise in der Versionskontrollhistorie sichtbar, wenn jemand hinschaut.

- **Beginnen Sie bei den jüngsten Schulden, nicht bei den alten.** Nehmen Sie die in den letzten sechs bis zwölf Monaten eingeführten Posten — identifizierbar aus dem Workaround-Register, Review-Kommentaren und Commit-Historie — und analysieren Sie diese. Schulden aus 2011 sagen Ihnen etwas über eine Organisation, die nicht mehr existiert.
- **Suchen Sie nach Häufung in Zeit, Ort und Umständen.** Auf ein Subsystem konzentrierte Schulden verweisen auf dessen Struktur oder Eigentümerschaft; auf die Wochen vor Releases konzentrierte Schulden verweisen auf den Release-Prozess; auf die Arbeit eines Teams konzentrierte Schulden verweisen auf Kompetenz oder Arbeitslast.
- **Fragen Sie, was der günstige Pfad im Moment der Entscheidung war.** Schulden sind fast immer die rational lokale Wahl unter den geltenden Einschränkungen. Die produktive Frage ist, was den guten Pfad teuer gemacht hat — keine Tests, keine Zeit, kein Wissen, keine Autorität, Nein zu sagen —, weil diese Einschränkung das eigentliche Ziel ist.
- **Nutzen Sie eine schuldfreie Technik.** Die Analyse benennt Mechanismen, nicht Personen. In dem Moment, in dem sie Einzelpersonen identifiziert, hört die Information auf zu fließen, und die Analyse wird wertlos, weil die Menschen, die wissen, warum die Abkürzung genommen wurde, diejenigen sind, die sie genommen haben.
- **Suchen Sie nach der fehlenden Fähigkeit.** Ein wiederkehrendes Muster ungetesteter Änderungen in einem Bereich bedeutet häufig, dass das Testen dieses Bereichs echt schwer ist, nicht dass Entwickler nachlässig sind. Der Eingriff ist dann eine Nahtstelle oder ein Test-Fixture, keine Ermahnung.
- **Prüfen Sie die Anreize.** Wenn Lieferung gemessen wird und Qualität nicht, ist Schuldenentstehung das vorhersehbare Ergebnis, und keine Menge an Prozess wird das beheben. Dieser Befund ist unangenehm und oft der eigentliche.
- **Quantifizieren Sie die Entstehungsrate**, wo möglich — pro Quartal hinzugefügte Workarounds, pro Release aufgezeichnete Abkürzungen. Eine Rate macht es möglich zu sagen, ob ein Eingriff funktioniert hat, und ohne eine ist die Verbesserung Ansichtssache.
- **Beheben Sie einen Mechanismus nach dem anderen und messen Sie neu.** Mehrere gleichzeitige Eingriffe machen es unmöglich zu wissen, welcher funktioniert hat, und das Wissen darüber, welche Mechanismen wichtig sind, ist der dauerhafte Ertrag.
- **Speisen Sie die Befunde in Retrospektiven ein**, statt einen Bericht zu erstellen. Das Team, das die Schulden erzeugt, ist das, das etwas ändern muss, und ein ihm überreichter Bericht wird das nicht bewirken.

## Tradeoffs ⇄

> Den Mechanismus zu beheben verhindert mehr Schulden, als Abbau entfernt, aber die Ursachen sind oft organisatorisch und außerhalb der Kontrolle des Teams zu ändern.

**Vorteile:**

- Die Entstehungsrate sinkt, was der einzige Weg ist, wie ein Abbau-Aufwand jemals vorankommt, statt auf der Stelle zu treten.
- Eingriffe zielen auf Einschränkungen statt auf Verhalten, was funktioniert — Menschen zu sagen, unter unveränderten Einschränkungen besseren Code zu schreiben, funktioniert zuverlässig nicht.
- Die Analyse deckt häufig eine fehlende Fähigkeit auf, deren Abwesenheit unsichtbar war, etwa einen Bereich, in dem Testen echt unpraktisch ist.
- Die Moral verbessert sich, wenn das Team die Rate sich ändern sieht, statt unbegrenzt gegen einen ungemessenen Zufluss abzubauen.
- Die Befunde sind meist günstig umzusetzen im Verhältnis zu den Schulden, die sie verhindern, da eine Prozess- oder Tooling-Korrektur klein ist im Vergleich zu Monaten von Abbau.

**Kosten und Risiken:**

- Die Ursachen sind häufig organisatorisch — Termindruck, Anreize, Ressourcenausstattung —, und sie zu benennen gibt dem Team nicht die Macht, sie zu ändern.
- Die Befunde können politisch unangenehm sein, besonders wenn die ehrliche Antwort ist, dass Managementdruck der Mechanismus ist.
- Die Zuordnung ist schwierig: Jetzt eingeführte Schulden werden möglicherweise erst in einem Jahr sichtbar, sodass die Analyse dem untersuchten Verhalten immer hinterherhinkt.
- Ohne einen schuldfreien Rahmen wird die Analyse schnell zu einem Audit, und die benötigte Information hört auf, verfügbar zu sein.
- Die Messung der Entstehungsrate erfordert die Erfassungsdisziplin — Workaround-Register, markierte Abkürzungen —, die möglicherweise noch nicht existiert, sodass die Messarbeit zuerst kommt.

## How It Could Be

Ein Team hatte drei Quartale lang einen Aufwand zur technischen-Schulden-Reduktion betrieben und konnte nicht nachweisen, dass sich die Situation verbessert hatte. Sie analysierten die in denselben drei Quartalen eingeführten Schulden und fanden 34 neue Posten gegenüber etwa 40 behobenen — sie hatten nahezu kostendeckend gearbeitet, ohne es zu wissen. Die Häufung war deutlich: 21 der 34 waren in den letzten zwei Wochen vor einem Release eingeführt worden, und 19 davon lagen in Codepfaden ohne Testabdeckung. Der Mechanismus war keine Nachlässigkeit. Ihr Release-Prozess konzentrierte Integrationsarbeit in ein zweiwöchiges Fenster, und in diesem Fenster kostete der sichere Pfad — einen Charakterisierungstest schreiben, bevor ungetesteter Code geändert wird — einen Tag, den niemand hatte. Zwei Eingriffe folgten: Continuous Integration, um die Arbeit zu verteilen, und eine kleine Bibliothek von Test-Fixtures für die drei unhandlichsten Bereiche. Die in den folgenden zwei Quartalen eingeführten Schulden sanken auf 11 Posten.

Der Anreizbefund kam aus derselben Analyse und war schwerer umzusetzen. Von den 13 Posten, die nicht mit dem Release-Fenster verbunden waren, gingen 9 auf eine einzige wiederkehrende Situation zurück: eine Anfrage, die direkt von einem leitenden Stakeholder an einen Entwickler kam, am Backlog vorbei, mit impliziter Dringlichkeit. Niemand hatte jemals eine abgelehnt. Die Engineering-Managerin des Teams brachte dies zu ihrem Direktor nicht als Beschwerde, sondern als gemessenen Befund — neun Schuldenposten in neun Monaten aus einem identifizierbaren Weg —, und das Ergebnis war eine Regel, die solche Anfragen über sie leitete. Die Regel wurde im folgenden Jahr zweimal gebrochen statt etwa monatlich, was das Team als Erfolg statt als Versagen der Regel betrachtete.
