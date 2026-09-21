---
title: Definition of Ready
description: Vereinbarung, was ein Arbeitspaket enthalten muss, bevor das Team damit
  beginnt, sodass halb spezifizierte Arbeit nicht mehr in die Entwicklung gelangt
  und dort steckenbleibt.
category:
- Requirements
- Process
- Team
problems:
- frequent-changes-to-requirements
- inadequate-requirements-gathering
- requirements-ambiguity
- poor-planning
- changing-project-scope
- scope-creep
- large-estimates-for-small-changes
- work-blocking
- eager-to-please-stakeholders
- incomplete-projects
- implementation-rework
- reduced-feature-quality
- constantly-shifting-deadlines
- delayed-project-timelines
- gold-plating
- missed-deadlines
- scope-change-resistance
- stakeholder-dissatisfaction
- time-pressure
- unrealistic-deadlines
- unrealistic-schedule
- feature-creep
- feature-factory
- large-feature-scope
- planning-dysfunction
- product-direction-chaos
- excessive-customization
- process-software-misfit
layout: solution
lang: de
en_slug: definition-of-ready
related_solutions:
- slug: definition-of-done
  similarity: 0.75
- slug: production-readiness-criteria
  similarity: 0.7
- slug: work-in-progress-limits
  similarity: 0.65
- slug: explicit-prioritization-framework
  similarity: 0.65
- slug: change-impact-analysis
  similarity: 0.65
- slug: evolutionary-requirements-development
  similarity: 0.65
---

## Description

Eine Definition of Ready ist eine vereinbarte Checkliste, die ein Arbeitspaket erfüllen muss, bevor sich das Team verpflichtet, damit zu beginnen: Das Problem ist formuliert, die Abnahmekriterien sind geschrieben, die betroffenen Systeme sind identifiziert, die Abhängigkeiten sind bekannt, und jemand ist verfügbar, um Fragen zu beantworten. Es ist das Eingangstor, das die Definition of Done am Ausgang spiegelt. Ihr Zweck ist es, das spezifische Versagen zu stoppen, bei dem Arbeit auf der Stärke einer einzeiligen Beschreibung in die Entwicklung gezogen wird, drei Tage später an einer Frage steckenbleibt, die niemand beantworten kann, und entweder blockiert liegen bleibt oder auf einer Annahme fortschreitet, die sich als falsch herausstellt. In Legacy-Kontexten verdient ein Eintrag besonderes Gewicht: welches bestehende Verhalten sich nicht ändern darf. Diese Frage ist während der Vorbereitung beantwortbar und enorm teuer, nachdem die Änderung gebaut wurde, zu beantworten.

## How to Apply ◆

> In einem Legacy-System ist das folgenreichste Unbekannte meist nicht, wie das neue Verhalten sein soll, sondern was vom aktuellen Verhalten abhängt — und genau das lässt eine übereilte Anforderung aus.

- Schreiben Sie die Checkliste **mit dem Team und wer auch immer die Arbeit liefert**, nicht für sie. Eine einem Product Owner auferlegte Definition of Ready wird zu einem Hindernis, das umgangen wird; eine mit ihm vereinbarte wird zu einem gemeinsamen Standard.
- Halten Sie sie auf **fünf bis acht Punkte** begrenzt. Eine lange Checkliste garantiert, dass nie etwas bereit ist, was dazu führt, dass sie aufgehoben wird, was das Team dort belässt, wo es begann, aber mit einem zusätzlichen Ritual.
- Verlangen Sie **Abnahmekriterien, ausgedrückt als beobachtbares Verhalten** — gegeben diese Situation, wenn dies geschieht, dann folgt dies. Kriterien, die nicht geprüft werden können, können nicht abgeschlossen werden, und Arbeit ohne Abschlussbedingung ist, wo Scope Creep eintritt.
- Beziehen Sie einen Punkt ein für **welches bestehende Verhalten erhalten bleiben muss**. Dies ist der Legacy-spezifische Eintrag, und er zahlt sich wiederholt aus: Er zwingt jemanden, die betroffenen Konsumenten zu identifizieren, bevor die Änderung gebaut wird, statt nachdem sie sie bricht.
- Verlangen Sie, dass **Abhängigkeiten und erforderlicher Zugriff identifiziert sind** — das andere Team, das etwas ändern muss, die benötigte Umgebung, die benötigten Daten, die Genehmigung, die nötig sein wird. Dies sind die Punkte, die zu mehrtägigen Blockaden werden, sobald die Arbeit begonnen hat.
- Benennen Sie, **wer Fragen zu diesem Punkt beantwortet**, und bestätigen Sie, dass diese Person während des Zeitraums, in dem die Arbeit geplant ist, tatsächlich verfügbar ist. Arbeit, deren einziger informierter Stakeholder im Urlaub ist, wird stocken, egal wie gut sie spezifiziert ist.
- Verlangen Sie, dass der Posten **klein genug ist, um innerhalb eines Zyklus abgeschlossen zu werden**. Ist er das nicht, ist das Aufteilen Teil des Bereitmachens, und das Aufteilen zeigt meist, dass manche Teile bereit sind und manche nicht.
- **Setzen Sie es beim Ziehen durch, nicht bei der Planung.** Arbeit, die die Definition nicht erfüllt, wird nicht begonnen; sie geht zur Vorbereitung zurück. Durchsetzung nur bei der Planung lässt Posten im Intervall verfallen.
- **Verfolgen Sie, wie oft Posten die Prüfung nicht bestehen und warum.** Ein konsistentes Scheitern an demselben Punkt — Abnahmekriterien oder betroffene Konsumenten — verweist auf eine spezifische Lücke darin, wie Arbeit vorgelagert vorbereitet wird, was nützlicher ist als die Checkliste selbst.

## Tradeoffs ⇄

> Ein Eingangstor verhindert, dass Arbeit mitten im Flug stockt, auf Kosten einer Warteschlange davor und eines echten Risikos, genutzt zu werden, um Arbeit abzulehnen statt sie vorzubereiten.

**Vorteile:**

- Arbeit hört auf zu stocken, nachdem sie begonnen hat, was die teure Art des Stockens ist: Kontext ist geladen, Kapazität ist gebunden, und der Posten belegt einen Slot, während gewartet wird.
- Anforderungsschwankung geht zurück, weil Mehrdeutigkeit vor der Implementierung gelöst wird statt während ihr entdeckt zu werden.
- Schätzungen verbessern sich, da ein Posten, der die Kriterien erfüllt, gut genug verstanden ist, um überhaupt geschätzt zu werden.
- Die Erhaltungsfrage bringt versteckte Konsumenten bestehenden Verhaltens früh ans Licht, wenn ihre Berücksichtigung eine Designentscheidung ist statt eines Notfalls.
- Vorbereitungsarbeit wird als echte Aktivität mit echten Kosten sichtbar, statt als etwas, das unsichtbar geschehen soll.

**Kosten und Risiken:**

- Sie erzeugt einen Rückstau nicht bereiter Arbeit, und wenn niemand Kapazität hat, Posten vorzubereiten, ist das Team blockiert mit dem Anschein, gut organisiert zu sein.
- Die Checkliste kann zu einer Waffe werden, um Arbeit abzulehnen, was die Beziehung zu Stakeholdern schädigt und die Praxis irgendwann von oben aufgehoben wird.
- Überspezifikation ist ein echtes Risiko: zu weit getrieben, wird eine Definition of Ready zu vorgelagerter Analyse und zerstört die Fähigkeit, während des Bauens zu lernen.
- Echt explorative Arbeit — die Untersuchung eines Defekts, ein Spike für Unbekanntes — kann eine kriteriumsbasierte Checkliste nicht erfüllen und braucht eine explizite Ausnahme, sonst wird die Ausnahme informell für alles genommen.
- Die Vorbereitung von Posten verbraucht die Zeit der Menschen, die meist am gefragtesten sind, und diese Kosten müssen geplant statt angenommen werden.

## How It Could Be

Ein Team, das ein öffentliches Nahverkehrs-Ticketsystem pflegte, fand heraus, dass etwa ein Drittel der begonnenen Posten innerhalb der ersten drei Tage stockte, immer aus einem von drei Gründen: Niemand wusste, welche nachgelagerten Systeme die zu ändernden Daten konsumierten, die Abnahmekriterien waren ein einzelner Satz, oder die einzige Person, die die Anforderung verstand, war nicht verfügbar. Sie schrieben eine sechsteilige Definition of Ready, die genau diese drei plus Abhängigkeiten, Größe und einen benannten Frage-Beantworter abdeckte. Im ersten Monat scheiterten elf von neunzehn vorgeschlagenen Posten an der Prüfung und gingen zurück. Der Product Owner erlebte dies zunächst als Behinderung. Bis zum dritten Monat war die Scheiterrate auf zwei von zwanzig gesunken, weil Posten anders vorbereitet wurden, und der Anteil begonnener Arbeit, die stockte, war von einem Drittel auf unter fünf Prozent gefallen.

Das Erhaltungskriterium erzeugte das klarste Einzelergebnis. Ein Posten zur Änderung, wie Rabattcodes validiert wurden, bestand jede andere Prüfung leicht. Die Beantwortung von „welches bestehende Verhalten darf sich nicht ändern" erforderte, dass jemand hinschaute, und der Blick fand vier Batch-Jobs und eine Partnerintegration, die die Validierungsergebnisse direkt lasen, wovon zwei niemand im Team bekannt waren. Die Änderung wurde neu gestaltet, um den alten Ausgabepfad intakt zu halten, und brauchte vier zusätzliche Tage. Die Schätzung des Teams, was dieselbe Änderung gekostet hätte, wäre sie ohne dieses Wissen ausgeliefert worden — basierend auf einem vergleichbaren Vorfall im Vorjahr — waren mehrere Wochen und eine Partnereskalation.
