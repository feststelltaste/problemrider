---
title: Explizites Priorisierungs-Framework
description: Etablierung einer einzigen priorisierten Liste mit festgelegten Kriterien
  und einem verantwortlichen Eigentümer, sodass Priorität einmal entschieden statt
  fortlaufend neu verhandelt wird.
category:
- Management
- Process
- Business
problems:
- competing-priorities
- priority-thrashing
- changing-project-scope
- short-term-focus
- feature-factory
- product-direction-chaos
- scope-change-resistance
- gold-plating
- work-blocking
- project-resource-constraints
- market-pressure
- constantly-shifting-deadlines
- reduced-predictability
- unclear-sharing-expectations
- decision-paralysis
- delayed-decision-making
- project-authority-vacuum
- uneven-work-flow
- budget-overruns
- cascade-delays
- context-switching-overhead
- deadline-pressure
- delayed-issue-resolution
- delayed-project-timelines
- incomplete-projects
- missed-deadlines
- overworked-teams
- poor-planning
- time-pressure
- unrealistic-deadlines
- unrealistic-schedule
- analysis-paralysis
- delayed-bug-fixes
- eager-to-please-stakeholders
- increased-stress-and-burnout
- planning-credibility-issues
- planning-dysfunction
- poor-project-control
- unclear-goals-and-priorities
- excessive-customization
layout: solution
lang: de
en_slug: explicit-prioritization-framework
related_solutions:
- slug: product-owner
  similarity: 0.7
- slug: technical-debt-backlog
  similarity: 0.7
- slug: cost-of-delay
  similarity: 0.7
- slug: decision-rights-and-escalation
  similarity: 0.7
- slug: clear-roles-and-ownership
  similarity: 0.65
- slug: sustainable-pace-practices
  similarity: 0.65
---

## Description

Ein explizites Priorisierungs-Framework ersetzt die implizite Verhandlung, die sonst darüber entscheidet, woran ein Team arbeitet. Es besteht aus drei Teilen: einer einzigen priorisierten Liste, in die jede Arbeit eintreten muss, schriftlichen Kriterien, nach denen sich die Position auf dieser Liste bestimmt, und einer benannten Person, die für die Rangfolge verantwortlich ist. Teams ohne dies haben keinen Mangel an Prioritäten — sie haben mehrere konkurrierende, jede gestützt von einem Stakeholder mit informellem Einfluss, weshalb die Priorität eines Elements davon abhängt, wer zuletzt am lautesten gefragt hat. Das Framework macht die Priorisierung nicht objektiv, und es sollte das auch nicht beanspruchen; sein Zweck ist es, den Zielkonflikt sichtbar zu machen. Wenn das Hinzufügen eines Elements bedeutet, benennen zu müssen, was es verdrängt, verschiebt sich das Gespräch von „kannst du das auch noch machen" zu „welches von diesen beiden ist wichtiger", was die einzige Version des Gesprächs ist, die konvergieren kann.

## How to Apply ◆

> In Legacy-Umgebungen muss die Liste Arbeit ohne sichtbaren Geschäftsertrag aufnehmen können — Migrationen, Abhängigkeits-Upgrades und Stabilisierung —, sonst verliert diese Arbeit immer gegen Features und wird weiterhin unsichtbar, nachts oder gar nicht erledigt.

- Etablieren Sie **eine Liste für alle Arbeit**, einschließlich Features, Defekte, Wartung, regulatorischer Verpflichtungen und Infrastruktur. Parallele Listen führen das ursprüngliche Problem wieder ein: Wer eine zweite Liste kontrolliert, kontrolliert einen zweiten Satz von Prioritäten, und das Team muss zwischen ihnen vermitteln.
- Schreiben Sie die **Kriterien** auf, nach denen Elemente eingestuft werden, in Reihenfolge ihrer Gewichtung. Ein praktikabler Satz für Legacy-Kontexte: regulatorische oder vertragliche Verpflichtung, Risiko eines unmittelbar bevorstehenden Ausfalls, direkte Umsatz- oder Kostenauswirkung, Kosten der Verzögerung und Aufwand. Veröffentlichen Sie sie, denn ungeschriebene Kriterien sind von Günstlingswirtschaft nicht zu unterscheiden.
- Benennen Sie **einen verantwortlichen Eigentümer** der Rangfolge. Ausschüsse erzeugen Kompromissreihenfolgen, die niemanden zufriedenstellen und still übergangen werden. Der Eigentümer konsultiert breit und entscheidet allein, und seine Entscheidungen sind über einen festgelegten Eskalationspfad anfechtbar statt über Nebenkanäle.
- Erzwingen Sie eine **strikte Reihenfolge, keine Buckets**. Prioritätsstufen sind ein bekanntes Versagensmuster: Alles, was wichtig ist, wird hohe Priorität, und das Team ist zurück beim eigenständigen Auswählen. Wenn zwei Elemente nicht gegeneinander eingestuft werden können, sind die Kriterien unvollständig und müssen verfeinert werden.
- Machen Sie **Verdrängung explizit**. Jede Ergänzung oberhalb der aktuellen Arbeitslinie muss benennen, was sie nach unten drückt. Diese eine Regel ist es, die einen unbegrenzten Strom dringender Anfragen in eine endliche Menge von Zielkonflikten umwandelt, und daher stammt der größte Teil des Werts des Frameworks.
- Setzen Sie einen **Takt für die Neueinstufung** — typischerweise wöchentlich oder pro Iteration — und halten Sie die Rangfolge zwischen diesen Punkten stabil, außer bei echten Notfällen, die im Voraus definiert werden. Prioritäts-Thrashing wird meist nicht durch sich ändernde Prioritäten verursacht, sondern durch deren kontinuierliche statt vereinbarte Änderung.
- Geben Sie **technischer Arbeit und Risikominderung einen verteidigten Anteil** an der Rangfolge. Weil solche Elemente bei jedem umsatzbasierten Kriterium schlecht abschneiden, werden sie nie organisch aufsteigen. Fügen Sie entweder ein explizites Risikokriterium mit echtem Gewicht hinzu oder reservieren Sie einen festen Kapazitätsanteil, der separat eingestuft wird.
- Erfassen Sie die **Kosten der Verzögerung** für Elemente, die immer wieder verschoben werden. Ein elfmal verschobenes Element ist eine Entscheidung, die die Organisation faktisch bereits getroffen hat; die aufgelaufenen Kosten sichtbar zu machen führt entweder dazu, dass es erledigt oder entfernt wird — beides besser als unbegrenzte Verschiebung.
- Veröffentlichen Sie die priorisierte Liste dort, wo Stakeholder sie sehen können, ohne fragen zu müssen. Die meisten Eskalationen sind eine Suche nach Information über die Position; Sichtbarkeit macht die Eskalation überflüssig.

## Tradeoffs ⇄

> Explizite Priorisierung verwandelt politischen Konflikt in sichtbare Zielkonflikte, was das Problem des Teams löst, aber die Schwierigkeit zu den Stakeholdern verschiebt — wo sie hingehört und wo ihr widerstanden werden wird.

**Vorteile:**

- Das Team hört auf, zwischen Stakeholdern zu vermitteln — eine Arbeit, für die es weder die Autorität noch die Information hat, und die eine wesentliche Quelle der Demoralisierung ist.
- Prioritäts-Thrashing sinkt, weil Neueinstufungen zu vereinbarten Zeitpunkten geschehen und eine erklärte Verdrängung erfordern statt eines Flurgesprächs.
- Verschobene Arbeit wird sichtbar statt zu verschwinden, was der einzige Weg ist, wie Wartungs- und Risikominderungselemente jemals nach ihren Verdiensten diskutiert werden.
- Die Planung wird glaubwürdiger, da Prognosen auf einer stabilen Reihenfolge basieren statt auf dem, was die Woche überlebt.
- Stakeholder, die eine Einstufungsentscheidung verlieren, können sehen, warum, was generell weit besser akzeptiert wird als eine unerklärte Verzögerung.

**Kosten und Risiken:**

- Das Framework erfordert echte Autorität hinter dem Eigentümer. Ohne sie wird die Liste zu einem Dokument, das Absichten beschreibt, während die eigentliche Arbeit anderswo bestimmt wird, und das Team pflegt zwei Realitäten.
- Schriftliche Kriterien laden zum Ausspielen ein. Stakeholder lernen, Anfragen in den Begriffen zu formulieren, die am höchsten bewerten, besonders bei Risiko und Compliance, und die Kriterien brauchen periodische Neukalibrierung.
- Strikte Reihenfolge ist echt schwer und verbraucht beachtliche Managementzeit, besonders anfangs, wenn ein großer bestehender Rückstand eingestuft werden muss.
- Zielkonflikte explizit zu machen bringt Konflikte an die Oberfläche, die zuvor vom Team durch längere Arbeitszeiten absorbiert wurden. Dies ist der beabsichtigte Effekt, kann aber organisatorisch so erlebt werden, als hätte das Framework den Konflikt verursacht.
- Aufwandsbasierte Kriterien bevorzugen kleine Elemente und können notwendige große Arbeit aushungern, sofern große Elemente nicht zerlegt oder mit geschützter Kapazität versehen werden.

## How It Could Be

Ein Team, das ein Schadensbearbeitungssystem pflegte, nahm Anfragen von vier Abteilungen entgegen, von denen jede glaubte, ihre Arbeit sei bereits vereinbart. Das Team verbrachte etwa einen Tag pro Woche mit Priorisierungsdiskussionen und lieferte trotzdem, was der hartnäckigste Stakeholder verlangte. Ihr Abteilungsleiter übernahm die Eigentümerschaft einer einzigen priorisierten Liste mit vier veröffentlichten Kriterien und einer Regel: Alles, was oberhalb der Linie eingefügt wird, muss benennen, was es verdrängt. Die ersten drei Wochen waren strittig — zwei Stakeholder eskalierten zum Direktor —, aber die Eskalationen betrafen die Rangfolge, nicht das Team, und beide wurden in je einem einzigen Meeting gelöst. Im zweiten Monat waren die Neupriorisierungen des Teams mitten in der Iteration von durchschnittlich fünf auf unter eine gefallen, und die beiden Stakeholder, die eskaliert hatten, berichteten von höherer Zufriedenheit, weil sie sehen konnten, wann ihre Elemente eintreffen würden.

Dieselbe Liste löste ein zweites Problem, das das Team nicht erwartet hatte anzugehen. Eine Datenbankmigration war zwei Jahre lang verschoben worden, immer gegen Feature-Arbeit verlierend. Unter den neuen Kriterien schnitt sie beim Risiko eines unmittelbar bevorstehenden Ausfalls hoch ab, war aber zu groß, um irgendwo hineinzupassen, sodass sie auf Position drei verweilte und nichts blockierte, während sie sichtbar unerledigt blieb. Ihre dauerhafte Präsenz nahe der Spitze der veröffentlichten Liste war es, was die Führung schließlich veranlasste, sie als separates Vorhaben zu finanzieren, statt zu erwarten, dass sie nebenbei absorbiert würde. Die Migration wurde im folgenden Quartal abgeschlossen, fünf Wochen bevor der Anbieter den Support für die alte Datenbankversion einstellte.
