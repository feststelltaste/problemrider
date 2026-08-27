---
title: Kriterien für Produktionsreife
description: Definition dessen, was eine Komponente bieten muss —
  Observability, Wiederherstellung, Ownership, Dokumentation — bevor sie
  Produktivlast tragen darf.
category:
- Operations
- Process
- Architecture
problems:
- rapid-prototyping-becoming-production
- immature-delivery-strategy
- quality-compromises
- monitoring-gaps
- operational-overhead
- inadequate-test-infrastructure
- constant-firefighting
- lack-of-ownership-and-accountability
- high-defect-rate-in-production
- log-spam
- insufficient-testing
- unclear-documentation-ownership
- environment-variable-issues
- no-formal-change-control-process
- database-connection-leaks
- inadequate-configuration-management
- incorrect-max-connection-pool-size
- legacy-configuration-management-chaos
- logging-configuration-issues
- misconfigured-connection-pools
- release-anxiety
- resource-allocation-failures
- service-discovery-failures
layout: solution
lang: de
en_slug: production-readiness-criteria
related_solutions:
- slug: definition-of-done
  similarity: 0.7
- slug: definition-of-ready
  similarity: 0.7
- slug: clear-roles-and-ownership
  similarity: 0.7
- slug: status-monitoring
  similarity: 0.7
- slug: observability-and-monitoring
  similarity: 0.65
- slug: clear-ownership-model
  similarity: 0.65
---

## Description

Kriterien für Produktionsreife sind eine explizite Checkliste, die eine Komponente erfüllen muss, bevor sie echte Nutzer bedienen darf: Sie kann beobachtet werden, sie kann wiederhergestellt werden, jemand besitzt sie, ihre Fehlermodi sind bekannt, und ihre Betriebsverfahren sind aufgeschrieben. Die Kriterien existieren wegen eines spezifischen und extrem häufigen Fehlerpfads — etwas wird schnell gebaut, um eine Idee zu demonstrieren, es funktioniert, es wird genutzt, und es ist in Produktion, bevor jemand entscheidet, dass es das sein sollte. Nichts daran war dafür entworfen, betrieben zu werden: Es gibt keine Metriken, keine Alarme, kein Runbook und oft keinen Eigentümer. Legacy-Systeme bestehen erheblich aus Komponenten, die auf diese Weise ankamen, und die betriebliche Last, die sie auferlegen, wird täglich von wem auch immer im Bereitschaftsdienst ist bezahlt. Die Kriterien verwandeln eine implizite Drift in eine explizite Entscheidung, die dann bewusst getroffen werden kann — einschließlich der Entscheidung, eine Lücke zu akzeptieren und sie aufzuzeichnen.

## How to Apply ◆

> Die Komponenten, die in einer Legacy-Landschaft am meisten schaden, sind normalerweise nicht diejenigen, die schlecht gebaut wurden, sondern diejenigen, die nie als dauerhaft gedacht waren und nie anschließend betriebsfähig gemacht wurden.

- **Schreiben Sie die Kriterien als kurze Checkliste**, die die Bereiche abdeckt, die Betriebsfähigkeit bestimmen: Observability, Fehlerverhalten, Wiederherstellung, Ownership, Abhängigkeiten und Dokumentation. Acht bis zwölf Punkte reichen. Eine längere Liste wird Punkt für Punkt heruntergehandelt, genau in dem Moment, in dem Druck besteht, auszuliefern.
- Fordern Sie **Observability vor dem Launch**: Die Komponente emittiert Metriken für ihre Schlüsseloperationen, ihre Logs sind strukturiert und korrelierbar, und es gibt mindestens einen Alarm, der auslöst, wenn sie ihre Aufgabe nicht erfüllt. Eine Komponente, die nur durch eine sich beschwerende Nutzer beobachtet werden kann, ist die teuerste Art zu betreiben.
- Fordern Sie **ein benanntes besitzendes Team**, keine Einzelperson. Eigentümerlose Komponenten sind diejenigen, die verfallen, und Ownership, die einer Person zugewiesen wird, die später geht, ist Ownership, die still verschwindet.
- Fordern Sie, dass **Fehlerverhalten bekannt und festgehalten ist**: was passiert, wenn jede Abhängigkeit nicht verfügbar ist, was das Timeout- und Wiederholungsverhalten ist, und ob Fehler elegant oder total sind. Diese Fragen vor dem Launch zu beantworten ändert üblicherweise das Design.
- Fordern Sie **einen Wiederherstellungspfad** — wie sie neu gestartet wird, wie sie zurückgerollt wird, und ob ihr Zustand wiederhergestellt werden kann. In einer Legacy-Landschaft ist es üblich, Komponenten ohne jegliches getestetes Wiederherstellungsverfahren zu finden, und die Entdeckung erfolgt immer zur schlimmsten Zeit.
- Fordern Sie ein **Runbook, das die bekannten Fehlermodi abdeckt**, geschrieben von wem auch immer sie gebaut hat. Zwei beim Launch geschriebene Seiten sind weit mehr wert als die während eines Vorfalls um drei Uhr morgens versuchte Rekonstruktion.
- **Wenden Sie die Kriterien auch auf bestehende Komponenten an**, rückwirkend und in Prioritätsreihenfolge. Die Checkliste gegen die Komponenten laufen zu lassen, die die meisten Vorfälle erzeugen, erklärt typischerweise den größten Teil der Vorfalllast an einem Nachmittag.
- Erlauben Sie **explizite, aufgezeichnete Ausnahmen** mit einem Eigentümer und einem Datum. Der Zweck der Kriterien ist, die Lücke zu einer Entscheidung statt zu einem Unfall zu machen; ein starres Gate ohne Ausnahmepfad wird vollständig umgangen und gilt dann für nichts.
- **Verifizieren statt behaupten.** Eine ohne Beweis abgehakte Checkliste misst Optimismus. Bitten Sie darum, das Dashboard zu sehen, den Alarm auszulösen, den Rollback in einer niedrigeren Umgebung auszuführen.

## Tradeoffs ⇄

> Die Kriterien verhindern die langsame Anhäufung nicht betriebsfähiger Komponenten, auf Kosten der Verlangsamung jedes Launches und der Notwendigkeit einer Autorität, die bereit ist, sie durchzusetzen.

**Vorteile:**

- Prototypen hören auf, standardmäßig Produktionssysteme zu werden, was der Ursprung eines großen Teils der Betriebslast in langlebigen Landschaften ist.
- Betriebliche Last sinkt messbar, weil Komponenten mit der Observability und Wiederherstellung ankommen, die sonst erst nach dem dritten Vorfall hinzugefügt würden, wenn überhaupt.
- Ownership wird zum Zeitpunkt der Erstellung etabliert, wenn offensichtlich ist, wer sie besitzt, statt Jahre später rekonstruiert zu werden, wenn es nicht mehr offensichtlich ist.
- Die rückwirkend angewendete Checkliste ist eine effiziente Diagnose für eine bestehende Landschaft und identifiziert schnell, woher die Vorfalllast kommt.
- Bereitschaftsdienst wird nachhaltiger, was direkte Auswirkung auf die Bindung derjenigen hat, die ihn tragen.

**Kosten und Risiken:**

- Jeder Launch dauert länger, und die Kosten konzentrieren sich auf kleine Komponenten, wo der Overhead proportional am größten ist.
- Durchsetzung erfordert Autorität. Kriterien, die von wem auch immer in Eile ist überstimmt werden können, liefern Dokumentation dessen, was hätte passieren sollen, und sonst nichts.
- Eine Checkliste fördert Compliance über Urteilsvermögen: Eine Komponente kann jeden Punkt erfüllen und trotzdem schlecht für den Betrieb entworfen sein.
- Einheitlich angewendet, erlegen die Kriterien einem internen Werkzeug dieselbe Last auf wie einem kundenseitigen Dienst, was unverhältnismäßig ist und Groll erzeugt.
- Rückwirkende Anwendung auf einen großen Legacy-Bestand offenbart mehr Lücken, als finanziert werden können, was ohne einen priorisierten Plan demoralisierend sein kann.

## How It Could Be

Ein Team erbte eine Landschaft aus etwa 40 Diensten, angesammelt über acht Jahre, von denen sie entdeckten, dass 14 überhaupt keine Alarmierung und 6 keinen identifizierbaren Eigentümer hatten. Ihre Bereitschaftsrotation lag bei durchschnittlich 11 Alarmierungen pro Woche, und ungefähr die Hälfte davon waren Vorfälle, die von Nutzern statt von Monitoring entdeckt wurden. Sie schrieben eine Zehn-Punkte-Reife-Checkliste und wendeten sie rückwirkend an, schlechteste zuerst nach Vorfallzahl. Die Übung selbst war aufschlussreich: Der einzige schlimmste Übeltäter war ein Währungsumrechnungsdienst, geschrieben als Zwei-Wochen-Prototyp 2019, immer noch laufend, ohne Metriken, ohne Runbook und mit einer fest codierten Wiederholungsschleife, die still Fehler schluckte. Die obersten acht Komponenten auf die Kriterien zu bringen dauerte ein Quartal. Alarmierungen fielen von 11 pro Woche auf 3, und der Anteil, der durch Monitoring statt durch Nutzer erkannt wurde, ging von etwa der Hälfte auf über neunzig Prozent.

Die vorausschauende Anwendung der Kriterien stoppte eine Wiederholung. Ein Team baute eine Demonstration eines automatisierten Dokumentenklassifizierungs-Features, das Stakeholder sofort mochten, und der Druck, es innerhalb von zwei Wochen echten Nutzern vorzulegen, war erheblich. Unter dem vorherigen Regime wäre es unverändert live gegangen. Der Reife-Review fand, dass es keine Fehlerbehandlung für die Nichtverfügbarkeit des Klassifizierungsdienstes, keine Metriken und keinen Eigentümer über die Person hinaus hatte, die es gebaut hatte. Das Team nahm sich neun zusätzliche Tage, um diese hinzuzufügen, und zeichnete zwei Ausnahmen mit Daten auf — kein Lasttest und kein automatisierter Rollback —, die im folgenden Monat geschlossen wurden. Der Klassifizierungsdienst war sechs Wochen später vier Stunden lang nicht verfügbar, und das Feature degradierte zu manuellem Routing mit einem Alarm, was die Demonstrationsversion durch Rückgabe von Fehlern an Nutzer ohne Benachrichtigung an irgendjemanden behandelt hätte.
