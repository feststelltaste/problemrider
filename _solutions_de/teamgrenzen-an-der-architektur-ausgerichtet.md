---
title: Teamgrenzen an der Architektur ausgerichtet
description: Ziehen von Teamgrenzen entlang der tatsächlichen
  Systemgrenzen, sodass die meisten Änderungen von einem Team ohne
  teamübergreifende Koordination abgeschlossen werden können.
category:
- Team
- Architecture
- Management
problems:
- organizational-structure-mismatch
- team-coordination-issues
- reduced-team-flexibility
- duplicated-work
- team-confusion
- communication-risk-within-project
- communication-risk-outside-project
- rapid-team-growth
- work-blocking
- shared-database
- team-silos
- shared-dependencies
- approval-dependencies
- duplicated-effort
- cascade-delays
- maintenance-bottlenecks
layout: solution
lang: de
en_slug: team-boundaries-aligned-to-architecture
related_solutions:
- slug: clear-roles-and-ownership
  similarity: 0.75
- slug: modularization-and-bounded-contexts
  similarity: 0.7
- slug: domain-aligned-architecture
  similarity: 0.7
- slug: clear-ownership-model
  similarity: 0.7
- slug: team-autonomy-and-empowerment
  similarity: 0.7
- slug: communities-of-practice
  similarity: 0.7
---

## Description

Die Ausrichtung von Teamgrenzen an der Architektur bedeutet, Teams um Teile des Systems herum zu organisieren, die unabhängig geändert werden können, sodass ein typisches Arbeitspaket innerhalb eines Teams abgeschlossen wird, statt über drei hinweg ausgehandelt zu werden. Die Beziehung zwischen Organisationsstruktur und Systemstruktur verläuft in beide Richtungen: Die Schnittstellen eines Systems kommen dazu, die Kommunikationsstruktur der Organisation zu spiegeln, die es gebaut hat, und umgekehrt wird ein Team, dessen Verantwortlichkeiten die echten Nähte des Systems durchschneiden, den Großteil seiner Kapazität für Koordination aufwenden. In Legacy-Kontexten ist die Diskrepanz üblicherweise geerbt statt gewählt — Teams wurden um Technologieschichten herum gebildet, um Projekte, die vor Jahren endeten, oder um Einzelpersonen, die seitdem gegangen sind. Die Korrektur besteht nicht immer darin, die Teams neu zu organisieren; wo die Nähte des Systems falsch sind, ist das Verschieben der Nähte manchmal der bessere Schritt, und zu wissen, welchen Hebel man ziehen soll, ist der Großteil der Arbeit.

## How to Apply ◆

> Legacy-Systeme haben selten saubere Nähte, an denen man sich ausrichten kann, daher ist dies üblicherweise eine zweiseitige Anstrengung: Teams bewegen sich auf die Architektur zu und die Architektur bewegt sich auf die Teams zu.

- **Kartieren Sie zunächst die aktuelle Realität**: Erfassen Sie für die letzten paar Monate abgeschlossener Arbeit, wie viele Teams jedes Element benötigte. Wenn ein großer Anteil der Elemente zwei oder mehr Teams benötigt, sind die Grenzen falsch ausgerichtet, und der Anteil ist die Messgröße, die es wert ist, verfolgt zu werden, während Änderungen vorgenommen werden.
- Identifizieren Sie die **tatsächlichen Nähte des Systems** statt seiner beabsichtigten. Zeitliche Kopplung in der Versionskontrollhistorie, gemeinsam genutzte Datenbanktabellen und die Schnittstellen, die am häufigsten gemeinsam geändert werden, offenbaren, wo das System genuin teilbar ist und wo nicht.
- Bevorzugen Sie **Grenzen um Geschäftsfähigkeiten herum** gegenüber Grenzen um Technologieschichten herum. Ein Frontend-Team, ein Backend-Team und ein Datenbank-Team garantieren, dass jede nutzersichtbare Änderung alle drei erfordert, was die häufigste und teuerste Form dieser Fehlausrichtung ist.
- Geben Sie jedem Bereich **ein verantwortliches Team**, und machen Sie es explizit. Gemeinsame Eigentümerschaft eines kritischen Moduls durch drei Teams produziert zuverlässig das Ergebnis, dass keines von ihnen es pflegt, und gemeinsame Eigentümerschaft der Datenbank ist die häufigste spezifische Instanz.
- Wo eine Naht nicht existiert, aber benötigt wird, **schaffen Sie sie bewusst** — eine Schnittstelle, eine Anti-Corruption-Layer, eine Schema-Eigentümerschaftsaufteilung — vor oder zusammen mit der Teamänderung. Die Reorganisation von Teams um eine Grenze herum, die im Code nicht existiert, produziert dieselben Koordinationskosten mit zusätzlicher Verwirrung darüber, wer verantwortlich ist.
- Unterscheiden Sie Teams, die **einen Teil des Systems bauen und betreiben**, von Teams, die **etwas bereitstellen, das andere Teams nutzen**. Die zweite Art sollte daran gemessen werden, wie gut sie andere befähigt, nicht an ihrer eigenen Ausgabe, und sie sollte explizit für Support-Arbeit ausgestattet sein, die sonst unsichtbar wäre.
- Halten Sie die Anzahl der **Teams, mit denen sich ein Team koordinieren muss, klein** — drei oder vier ist eine praktikable Obergrenze. Darüber hinaus verbraucht Koordination den Großteil der Teamkapazität, egal wie gut die Meetings geführt werden.
- **Ändern Sie Grenzen selten und bewusst.** Jede Reorganisation kostet Monate an Kontextaufbau in einem Legacy-System, wo das Wissen über ein Subsystem die knappe Ressource ist. Zwei schlecht durchdachte Reorganisationen sind schlimmer als eine unvollkommene Struktur, die an Ort und Stelle bleibt.
- Verfolgen Sie den **teamübergreifenden Elementanteil nach der Änderung**. Wenn er nicht gesunken ist, sind auch die neuen Grenzen falsch ausgerichtet, und die Analyse der Systemnähte war falsch.

## Tradeoffs ⇄

> Ausrichtung reduziert Koordinationskosten erheblich, ist aber teuer zu erreichen, störend für Wissen und erfordert eine Architektur, die tatsächlich geteilt werden kann.

**Vorteile:**

- Der Großteil der Arbeit wird innerhalb eines einzelnen Teams abgeschlossen, was den Warte-, Verhandlungs- und Planungs-Overhead beseitigt, der die Zykluszeit in fehlausgerichteten Organisationen dominiert.
- Eigentümerschaft wird eindeutig, sodass Module nicht mehr zwischen Teams fallen — der übliche Ursprung des Codes, den niemand pflegt und niemand anzufassen wagt.
- Teams sammeln tiefes Wissen über ihren Bereich an, was in Legacy-Systemen überproportional zählt, wo Verständnis der limitierende Faktor ist.
- Doppelte Arbeit sinkt, weil die Grenze klar macht, welches Team für eine Fähigkeit verantwortlich ist, statt mehrere dasselbe Problem separat lösen zu lassen.
- Die Organisation kann wachsen, indem Teams an bestehenden Nähten hinzugefügt werden, statt Teams über den Punkt hinaus zu vergrößern, an dem sie effektiv koordinieren.

**Kosten und Risiken:**

- Reorganisationen zerstören angesammelten Kontext. In einem System, in dem es sechs Monate braucht, um in einem Subsystem produktiv zu werden, ist das Versetzen von Personen teuer auf eine Weise, die auf einem Organigramm unsichtbar ist.
- Viele Legacy-Systeme haben keine sauberen Nähte, an denen man sich ausrichten kann, sodass Ausrichtung zunächst architektonische Arbeit erfordert, die Quartale dauern kann und möglicherweise nicht finanziert wird.
- Starke Grenzen schaffen Silos, wenn nichts dem entgegenwirkt. Teamübergreifender Wissensaustausch, Rotation und gemeinsame Standards müssen bewusst gepflegt werden.
- Die Ausrichtung von Teams an der aktuellen Architektur zementiert diese Architektur, da die Organisation sich anschließend gegen Änderungen wehren wird, die die neuen Grenzen durchschneiden.
- Spezialisten — ein einsamer Mainframe-Experte, die einzige Person, die die Preis-Engine versteht — verteilen sich nicht ordentlich über fähigkeitsausgerichtete Teams, und die resultierenden einzelnen Abhängigkeitspunkte brauchen separate Behandlung.

## How It Could Be

Ein Versicherer hatte drei nach Technologie organisierte Teams: ein Web-Team, ein Service-Team und ein Mainframe-Team. Jede Policenänderung — das häufigste Arbeitspaket der Organisation — erforderte alle drei, mit durchschnittlich elf Tagen Wartezeit auf ein anderes Team. Sie maßen es und fanden heraus, dass 84 Prozent der abgeschlossenen Elemente mindestens zwei Teams durchquert hatten. Statt sofort zu reorganisieren, verbrachten sie zwei Quartale damit, Service-Schnittstellen um drei Geschäftsfähigkeiten herum zu bauen: Policenverwaltung, Schäden und Abrechnung. Erst dann bildeten sie drei Fähigkeitsteams, jedes mit Web-, Service- und Mainframe-Fähigkeiten. Sechs Monate nach der Änderung wurden 61 Prozent der Elemente innerhalb eines einzelnen Teams abgeschlossen, und die mittlere Zykluszeit war von 23 auf 9 Tage gesunken.

Eine zweite Organisation fand die gegenteilige Antwort. Ihre vier Produktteams schrieben alle direkt in ein gemeinsam genutztes Datenbankschema, und jede Schemaänderung erforderte ein Koordinationsmeeting mit allen vier. Eine Reorganisation der Teams hätte nicht geholfen, weil die Kopplung in den Daten statt im Code lag. Sie behielten die Teamstruktur bei und wiesen stattdessen die Eigentümerschaft jeder Tabellengruppe genau einem Team zu, wobei die anderen sie über die Schnittstelle dieses Teams zugreifen mussten. Die Migration dauerte drei Quartale und war unspektakulär, aber das Koordinationsmeeting wurde aufgelöst, und Schemaänderungen, die zuvor sechs Wochen brauchten, landeten nun innerhalb eines Sprints.
